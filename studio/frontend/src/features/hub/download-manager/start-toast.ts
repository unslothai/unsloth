// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The single start toast; its id derives from the job key so finalize() can dismiss it.

import { toast } from "@/lib/toast";

import { DOWNLOAD_KIND, type DownloadKind } from "./constants";

import type { CallerToast } from "./download-manager-types";
import {
  XET_NOTICE_DESCRIPTION_CLASS,
  XET_NOTICE_DURATION_MS,
} from "./xet-progress-notice";

export function startToastId(jobKey: string): string {
  return `download-start:${jobKey}`;
}

// Kind-scoped so a Chat model pick cannot erase an unrelated dataset notice on /hub.
const liveStartToasts = new Map<
  string,
  { route: string; kind: DownloadKind }
>();

let modelSelectionEpoch = 0;

function downloadKindOfJobKey(jobKey: string): DownloadKind {
  return jobKey.startsWith(`${DOWNLOAD_KIND.DATASET}:`)
    ? DOWNLOAD_KIND.DATASET
    : DOWNLOAD_KIND.MODEL;
}

/** Captured at start, since the preflight round trips can outlive the route. */
export function currentRoute(): string {
  return typeof window === "undefined" ? "" : window.location.pathname;
}

export function currentStartToastSelectionEpoch(): number {
  return modelSelectionEpoch;
}

export function showStartToast(
  jobKey: string,
  message: { title: string; description: string },
  originRoute: string = currentRoute(),
  originSelectionEpoch: number = currentStartToastSelectionEpoch(),
): void {
  const kind = downloadKindOfJobKey(jobKey);
  // Raised late: the sweep already ran. Dataset notices ignore Chat's selection epoch.
  if (
    originRoute !== currentRoute() ||
    (kind === DOWNLOAD_KIND.MODEL &&
      originSelectionEpoch !== currentStartToastSelectionEpoch())
  ) {
    return;
  }
  liveStartToasts.set(startToastId(jobKey), { route: originRoute, kind });
  toast.info(message.title, {
    id: startToastId(jobKey),
    description: message.description,
    duration: XET_NOTICE_DURATION_MS,
    classNames: { description: XET_NOTICE_DESCRIPTION_CLASS },
  });
}

export function liveCallerToast(
  message: CallerToast | undefined,
): CallerToast | undefined {
  if (!message) return undefined;
  return (message.stillValid?.() ?? true) ? message : undefined;
}

/** Shown only when the notice is not carrying the caller's message. */
export function showCallerToast(
  jobKey: string,
  message: CallerToast | undefined,
  originRoute?: string,
  originSelectionEpoch?: number,
): void {
  if (!message || message.noticeOnly) return;
  showStartToast(jobKey, message, originRoute, originSelectionEpoch);
}

/** Safe for a job that never raised one: sonner ignores an unknown id. */
export function dismissStartToast(jobKey: string): void {
  const id = startToastId(jobKey);
  liveStartToasts.delete(id);
  toast.dismiss(id);
}

/** Drop toasts for the surface just left, or chat's 8s toast covers the hub toolbar. */
export function dismissStartToasts(): void {
  const here = currentRoute();
  for (const [id, context] of liveStartToasts) {
    if (context.route === here) continue;
    liveStartToasts.delete(id);
    toast.dismiss(id);
  }
}

export function dismissStartToastsForModelSelection(): void {
  modelSelectionEpoch += 1;
  for (const [id, context] of liveStartToasts) {
    if (context.kind !== DOWNLOAD_KIND.MODEL) continue;
    liveStartToasts.delete(id);
    toast.dismiss(id);
  }
}
