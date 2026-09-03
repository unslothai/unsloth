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

// The cap and the count live on the server (utils/xet_notice_settings.py); a copy
// here could only drift out of step with what is enforced.

// Longer than the Toaster's 5s default, like the explanatory toasts in chat.
export const XET_NOTICE_DURATION_MS = 8000;

// SHORT on purpose, as a correctness constraint. At 62 + 330 chars sonner rendered
// this 235px tall over the Model hub toolbar, leaving 4 to 6 controls unclickable for
// 8s, which is what #9293 reverted. It must end above the filter row, near 158px, so
// title plus about two lines. Re-measure before adding a sentence.
export const XET_NOTICE_TITLE = "Download is running";
// The "switch to HTTP in Model Hub" advice is gone on purpose: it cost two lines and
// left the toast bottom at y=126.5 against a row centred at y=127, a margin any font
// or translation would have erased. Without it the toast ends near y=100.
export const XET_NOTICE_DESCRIPTION =
  "Xet sends the file in small pieces, so the bar can sit at 0% and then jump to done. Nothing is stuck.";
export const XET_NOTICE_DESCRIPTION_CLASS = "!text-muted-foreground";

/** The notice, plus whatever the starting surface wanted to add. Chat auto-loads and
 * says so; the Hub does not, passes nothing, and gets the short form. The test budget
 * applies to that form alone, since only it renders over the hub toolbar. */
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
 * No "the caller already toasted" clause: suppressing the notice whenever chat had
 * something to say removed the 0%-explanation where it is most useful. The caller's
 * line is folded in by composeNoticeDescription instead. */
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
