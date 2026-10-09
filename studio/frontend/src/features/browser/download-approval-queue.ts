// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import { getLocale, translate } from "@/i18n";
import { toast } from "@/lib/toast";
import { create } from "zustand";
import { hostOf, isWebUrl, safeDownloadName } from "./address";
import { isDangerousDownload } from "./download-safety";
import { useDownloadSitesStore } from "./download-sites-store";
import { useBrowserPrefsStore } from "./prefs-store";

/** `dangerous`: the file runs code when opened (download-safety.ts), so it asks whatever was remembered. */
export type DownloadRequest = {
  origin: string;
  label: string;
  name: string;
  dangerous: boolean;
  resolve: (allow: boolean) => void;
};

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
  const shown = safeDownloadName(name);
  const dangerous = isDangerousDownload(shown);
  const origin = downloadSiteOf(site);
  const label = (origin && hostOf(origin)) || hostOf(url) || url.slice(0, 80);
  const remembered = origin ? useDownloadSitesStore.getState().sites[origin] : undefined;
  if (remembered === "block") {
    toast.error(translate("browser.downloadPrompt.blocked", { host: label }, getLocale()));
    return Promise.resolve(false);
  }
  // A file that runs code always asks: a site's Always allow, or asking turned off, doesn't cover it.
  if (!dangerous && (remembered === "allow" || !useBrowserPrefsStore.getState().askBeforeDownloading)) {
    return Promise.resolve(true);
  }
  return ask(origin, label, shown, dangerous);
}

/** Picked from the context menu: no site answer applies, and only a file that runs code asks (with nothing to remember). */
export function approveChosenDownload(url: string, name: string): Promise<boolean> {
  const shown = safeDownloadName(name);
  if (!isDangerousDownload(shown)) return Promise.resolve(true);
  return ask("", hostOf(url) || url.slice(0, 80), shown, true);
}

function ask(origin: string, label: string, name: string, dangerous: boolean): Promise<boolean> {
  return new Promise((resolve) =>
    useApprovalStore.setState((state) => ({
      queue: [...state.queue, { origin, label, name, dangerous, resolve }],
    })),
  );
}

/** Ignored unless it's the download being asked about. Remembered, it also settles the site's other waiting downloads. */
export function answerDownload(request: DownloadRequest, allow: boolean, remember: boolean): void {
  const { queue } = useApprovalStore.getState();
  if (queue[0] !== request) return;
  const kept = remember && request.origin !== "";
  if (kept) useDownloadSitesStore.getState().setSite(request.origin, allow ? "allow" : "block");
  // A remembered Allow settles the site's other waiting files, except ones that run code: each of those asks.
  const answered = kept
    ? queue.filter((other) => other === request || (other.origin === request.origin && !(allow && other.dangerous)))
    : [request];
  useApprovalStore.setState({ queue: queue.filter((other) => !answered.includes(other)) });
  for (const other of answered) other.resolve(allow);
}
