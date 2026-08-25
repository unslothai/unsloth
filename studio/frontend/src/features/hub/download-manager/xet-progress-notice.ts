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

// The 3-use cap and the count both live on the server now (see
// studio/backend/utils/xet_notice_settings.py). A copy here could only ever drift
// out of step with the value actually being enforced.

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

/** The notice, plus whatever the starting surface wanted to say about this download.
 *
 * Chat's picker auto-loads the model when the transfer finishes and wants to say so;
 * the Hub does not auto-load anything, passes nothing, and gets the short form. That
 * split is why the caller supplies the sentence rather than this module hard-coding
 * it: on the Hub "it'll load automatically" would simply be false.
 *
 * The budget in the tests applies to XET_NOTICE_DESCRIPTION alone for the same
 * reason. The short form is the one that renders over the Model hub toolbar, which is
 * what #9293 reverted; the composed form only ever appears on chat, where there is no
 * toolbar underneath it. */
export function composeNoticeDescription(
  callerToast?: { description: string } | null,
): string {
  const extra = callerToast?.description?.trim();
  return extra ? `${XET_NOTICE_DESCRIPTION} ${extra}` : XET_NOTICE_DESCRIPTION;
}

/** Only the transport that behaves this way, and only while it is news.
 *
 * A start that attached to a job another tab or client already owns is
 * accepted and reports that job's transport, but this user started nothing.
 * `live` is the backend calling the job running with no cancel pending: a
 * start the user already cancelled has nothing to reassure them about.
 * Neither shows the notice nor spends one of the three.
 *
 * There is deliberately no "the caller already toasted" clause here any more. An
 * earlier attempt suppressed the notice whenever chat had something to say, which
 * removed the 0%-explanation exactly where a big first download makes it most
 * useful. The caller's line is folded into the notice instead, by
 * composeNoticeDescription, so nothing has to be dropped to avoid a second toast. */
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
