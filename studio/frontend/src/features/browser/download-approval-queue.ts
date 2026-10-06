// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Downloads waiting for "Download this file?" (DownloadApprovalDialog asks about each in turn).

import { getLocale, translate } from "@/i18n";
import { toast } from "@/lib/toast";
import { create } from "zustand";
import { hostOf, isWebUrl } from "./address";
import { useDownloadSitesStore } from "./download-sites-store";
import { useBrowserPrefsStore } from "./prefs-store";

/** `origin` is the site the answer is remembered for, "" when there is none; `label` names it. */
export type DownloadRequest = { origin: string; label: string; name: string; resolve: (allow: boolean) => void };

export const useApprovalStore = create<{ queue: DownloadRequest[] }>(() => ({ queue: [] }));

/** The origin a remembered answer is kept under: scheme, host and port, as browsers scope site
 *  permissions. A shortened host would let `https://www.example.com`'s answer cover
 *  `http://example.com:8443`, which can be another service. A blob: URL counts as the page that
 *  made it (its origin, per the URL Standard). "" for anything else: data:, about:, opaque blobs. */
export function downloadSiteOf(site: string): string {
  if (!isWebUrl(site) && !/^blob:/i.test(site)) return "";
  try {
    const { origin } = new URL(site);
    return isWebUrl(origin) ? origin : "";
  } catch {
    return "";
  }
}

/** Whether a file from `url` may be saved: a remembered answer for its site, else the user's.
 *  `site` is the page that started it, which the answer is kept for (the file's own address by
 *  default). Only a web page counts as a site: blob: and data: URLs have no host, and one answer
 *  for "" would then cover them on every site. */
export function approveDownload(url: string, name: string, site: string = url): Promise<boolean> {
  const origin = downloadSiteOf(site);
  const label = (origin && hostOf(origin)) || hostOf(url) || url.slice(0, 80);
  const remembered = origin ? useDownloadSitesStore.getState().sites[origin] : undefined;
  if (remembered === "block") {
    toast.error(translate("browser.downloadPrompt.blocked", { host: label }, getLocale()));
    return Promise.resolve(false);
  }
  if (remembered === "allow" || !useBrowserPrefsStore.getState().askBeforeDownloading) return Promise.resolve(true);
  return new Promise((resolve) =>
    useApprovalStore.setState((state) => ({ queue: [...state.queue, { origin, label, name, resolve }] })),
  );
}

/** Answer the download being asked about; ignored for any other (a dialog closing after its
 *  button answered). Remembered, the answer also covers the site's other downloads still
 *  waiting, which read the old answer before this one was given. */
export function answerDownload(request: DownloadRequest, allow: boolean, remember: boolean): void {
  const { queue } = useApprovalStore.getState();
  if (queue[0] !== request) return;
  const kept = remember && request.origin !== "";
  if (kept) useDownloadSitesStore.getState().setSite(request.origin, allow ? "allow" : "block");
  const answered = kept ? queue.filter((other) => other.origin === request.origin) : [request];
  useApprovalStore.setState({ queue: queue.filter((other) => !answered.includes(other)) });
  for (const other of answered) other.resolve(allow);
}
