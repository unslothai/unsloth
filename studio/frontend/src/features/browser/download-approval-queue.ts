// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import { getLocale, translate } from "@/i18n";
import { toast } from "@/lib/toast";
import { create } from "zustand";
import { hostOf, isWebUrl } from "./address";
import { useDownloadSitesStore } from "./download-sites-store";
import { useBrowserPrefsStore } from "./prefs-store";

export type DownloadRequest = { origin: string; label: string; name: string; resolve: (allow: boolean) => void };

export const useApprovalStore = create<{ queue: DownloadRequest[] }>(() => ({ queue: [] }));

/** Origin (scheme, host, port) an answer is kept under, as browsers scope permissions; a blob: URL counts as its creator's origin. "" for data:, about:, opaque blobs. */
export function downloadSiteOf(site: string): string {
  if (!isWebUrl(site) && !/^blob:/i.test(site)) return "";
  try {
    const { origin } = new URL(site);
    return isWebUrl(origin) ? origin : "";
  } catch {
    return "";
  }
}

/** Remembered answer for `site` (default: the file's address), else ask. Only web origins count: one answer for "" would cover blob:/data: everywhere. */
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

/** Ignored unless it's the download being asked about. Remembered, it also settles the site's other waiting downloads. */
export function answerDownload(request: DownloadRequest, allow: boolean, remember: boolean): void {
  const { queue } = useApprovalStore.getState();
  if (queue[0] !== request) return;
  const kept = remember && request.origin !== "";
  if (kept) useDownloadSitesStore.getState().setSite(request.origin, allow ? "allow" : "block");
  const answered = kept ? queue.filter((other) => other.origin === request.origin) : [request];
  useApprovalStore.setState({ queue: queue.filter((other) => !answered.includes(other)) });
  for (const other of answered) other.resolve(allow);
}
