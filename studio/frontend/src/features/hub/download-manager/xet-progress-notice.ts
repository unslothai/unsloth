// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Xet writes chunks out of order, so progress reads 0% then completes at once.

import {
  DOWNLOAD_KIND,
  type DownloadKind,
  type ResolvedTransport,
  TRANSPORT,
} from "./constants";

// The cap and count live on the server (utils/xet_notice_settings.py).

export const XET_NOTICE_DURATION_MS = 8000;

// Kept short on purpose: a taller toast covers the Model hub toolbar controls (must end above
// ~158px). Re-measure before adding a sentence.
export const XET_NOTICE_TITLE = "Download is running";
export const XET_NOTICE_DESCRIPTION =
  "Xet sends the file in small pieces, so the bar can sit at 0% and then jump to done. Nothing is stuck.";
export const XET_NOTICE_DESCRIPTION_CLASS = "!text-muted-foreground";
// Kept here so poll-loop can combine it with the Xet notice instead of a separate toast.
export const RESTART_NOTICE_TITLE = "Restarting this download";
export const RESTART_NOTICE_DESCRIPTION =
  "The earlier partial can't be resumed, so this download is starting over.";
export const RESTART_XET_NOTICE_DESCRIPTION =
  "The partial can't be resumed, so Xet is starting over. The bar may stay at 0% and jump to done.";

/** The size budget applies only to the short Hub form, which renders over the toolbar. */
function appendCallerDescription(
  description: string,
  callerToast?: { description: string } | null,
): string {
  const extra = callerToast?.description?.trim();
  return extra ? `${description} ${extra}` : description;
}

export function composeNoticeDescription(
  callerToast?: { description: string } | null,
): string {
  return appendCallerDescription(XET_NOTICE_DESCRIPTION, callerToast);
}

export function composeRestartNoticeDescription({
  xet,
  callerToast,
}: {
  xet: boolean;
  callerToast?: { description: string } | null;
}): string {
  return appendCallerDescription(
    xet ? RESTART_XET_NOTICE_DESCRIPTION : RESTART_NOTICE_DESCRIPTION,
    callerToast,
  );
}

/** Skip for attached jobs (the user started nothing) and for already-cancelled starts. */
export function shouldShowXetNotice(args: {
  kind: DownloadKind;
  transport: ResolvedTransport;
  attached: boolean;
  live: boolean;
}): boolean {
  return (
    args.kind === DOWNLOAD_KIND.MODEL &&
    args.transport === TRANSPORT.XET &&
    !args.attached &&
    args.live
  );
}
