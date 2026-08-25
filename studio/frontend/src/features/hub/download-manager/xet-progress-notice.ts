// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Xet writes chunks out of order, so progress reads 0% and then completes at
// once, which looks like a hang. No toast import here: poll-loop shows it.

import {
  DOWNLOAD_KIND,
  type DownloadKind,
  type ResolvedTransport,
  TRANSPORT,
} from "./constants";

export const XET_NOTICE_LIMIT = 3;
export const XET_NOTICE_STORAGE_KEY = "unsloth.studio.xetNoticeCount";

// Longer than the Toaster's 5s default, like the explanatory toasts in chat.
export const XET_NOTICE_DURATION_MS = 8000;

// Kept SHORT on purpose, and this is a correctness constraint rather than a
// style preference. The first version of this notice ran 62 characters of
// title and 330 of description, which sonner rendered 235px tall in the
// top-right corner. That is where the Model hub keeps its own toolbar, so for
// the 8s the toast was up it sat on top of the capability filter, the sort
// dropdown, the Models and Datasets tabs and the repo action icons: measured
// by hit testing each control's own centre point, 4 to 6 of them resolved into
// the toast and could not be clicked. That is what got #9159 reverted in
// #9293. The toast has to end above the filter row to block nothing, which
// means roughly 158px, so title plus about two lines of description. Adding a
// sentence here is not free: re-measure before you do.
export const XET_NOTICE_TITLE = "Download is running";
// The "switch to HTTP in Model Hub" advice from the first version is gone on
// purpose. Measured, it cost two rendered lines and put the toast's bottom edge
// at y=126.5 with the hub filter row's centre at y=127: a 0.5px margin, which
// is not a margin. Any font, zoom level or translation longer than the English
// would have pushed it back over and re-broken the toolbar. Without it the
// toast ends around y=100 and stops intersecting the row at all. The transport
// control is two clicks away and discoverable; a toast that eats the toolbar is
// not worth the shortcut.
export const XET_NOTICE_DESCRIPTION =
  "Xet sends the file in small pieces, so the bar can sit at 0% and then jump to done. Nothing is stuck.";
export const XET_NOTICE_DESCRIPTION_CLASS = "!text-muted-foreground";

// Carries the count when the write fails (private mode, quota), which would
// otherwise repeat the toast on every download.
let sessionShown = 0;

function readStoredCount(): number {
  if (typeof window === "undefined") return 0;
  try {
    const parsed = Number.parseInt(
      window.localStorage.getItem(XET_NOTICE_STORAGE_KEY) ?? "",
      10,
    );
    return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : 0;
  } catch {
    return 0;
  }
}

export function xetNoticesShown(): number {
  return Math.max(sessionShown, readStoredCount());
}

export function recordXetNoticeShown(): void {
  const next = xetNoticesShown() + 1;
  sessionShown = next;
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(XET_NOTICE_STORAGE_KEY, String(next));
  } catch {
    // Counted in memory above, so this session still stops at the limit.
  }
}

/** Take one of the three, or report that none are left.
 *
 * localStorage has no compare-and-set, so two tabs starting a download at the
 * same moment can both read the same count and both show the toast. Web Locks
 * are cross-tab, so the read and the write happen as one. Browsers without
 * them fall back to the plain check, which is the race above and still bounded
 * at one extra toast. */
export async function reserveXetNotice(): Promise<boolean> {
  const take = () => {
    if (xetNoticesShown() >= XET_NOTICE_LIMIT) return false;
    recordXetNoticeShown();
    return true;
  };
  const locks = globalThis.navigator?.locks;
  if (!locks) return take();
  try {
    return await locks.request(XET_NOTICE_STORAGE_KEY, take);
  } catch {
    return take();
  }
}

/** Only the transport that behaves this way, and only while it is news.
 *
 * A start that attached to a job another tab or client already owns is
 * accepted and reports that job's transport, but this user started nothing.
 * `live` is the backend calling the job running with no cancel pending: a
 * start the user already cancelled has nothing to reassure them about.
 * Neither shows the notice nor spends one of the three. */
export function shouldShowXetNotice(args: {
  kind: DownloadKind;
  transport: ResolvedTransport;
  attached: boolean;
  live: boolean;
  shown: number;
}): boolean {
  return (
    args.kind === DOWNLOAD_KIND.MODEL &&
    args.transport === TRANSPORT.XET &&
    !args.attached &&
    args.live &&
    args.shown < XET_NOTICE_LIMIT
  );
}
