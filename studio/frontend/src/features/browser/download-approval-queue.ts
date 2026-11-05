// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import { getLocale, translate } from "@/i18n";
import { toast } from "@/lib/toast";
import { create } from "zustand";
import { hostOf, isWebUrl, safeDownloadName } from "./address";
import type { DownloadsRule } from "./agent/permissions";
import { isDangerousDownload } from "./download-safety";
import { useDownloadSitesStore } from "./download-sites-store";
import { useBrowserPrefsStore } from "./prefs-store";

/** `dangerous`: the file runs code when opened (download-safety.ts), so it asks whatever was remembered. */
export type DownloadRequest = {
  origin: string;
  label: string;
  name: string;
  dangerous: boolean;
  /** From the browser agent's tab: asked under its own rules, so another file's answer never settles it. */
  agent: boolean;
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

/** Remembered answer for `site` (default: the file's address), else ask. Only web origins count: one answer for "" would cover blob:/data: everywhere.
 *  `agentRule`: the download came from the browser agent's tab, under the user's Agent permissions. */
export function approveDownload(
  url: string,
  name: string,
  site: string = url,
  agentRule?: DownloadsRule | null,
): Promise<boolean> {
  const shown = safeDownloadName(name);
  const dangerous = isDangerousDownload(shown);
  const origin = downloadSiteOf(site);
  const label = (origin && hostOf(origin)) || hostOf(url) || url.slice(0, 80);
  const remembered = origin ? useDownloadSitesStore.getState().sites[origin] : undefined;
  if (remembered === "block" || agentRule === "block") {
    toast.error(translate("browser.downloadPrompt.blocked", { host: label }, getLocale()));
    return Promise.resolve(false);
  }
  const prefs = useBrowserPrefsStore.getState();
  // A file that runs code asks unless the user allowed those in particular, never on the agent's say.
  const allowed = dangerous
    ? agentRule == null && (remembered === "allowAll" || !prefs.askBeforeProgramDownloads)
    : agentRule === "allow" ||
      (agentRule !== "ask" && (remembered === "allow" || remembered === "allowAll" || !prefs.askBeforeDownloading));
  if (allowed) return Promise.resolve(true);
  return ask(origin, label, shown, dangerous, agentRule != null);
}

/** Picked from the context menu: no site answer applies, and only a file that runs code asks, unless those were allowed everywhere. */
export function approveChosenDownload(url: string, name: string): Promise<boolean> {
  const shown = safeDownloadName(name);
  if (!isDangerousDownload(shown) || !useBrowserPrefsStore.getState().askBeforeProgramDownloads) return Promise.resolve(true);
  return ask("", hostOf(url) || url.slice(0, 80), shown, true);
}

function ask(origin: string, label: string, name: string, dangerous: boolean, agent = false): Promise<boolean> {
  return new Promise((resolve) =>
    useApprovalStore.setState((state) => ({
      queue: [...state.queue, { origin, label, name, dangerous, agent, resolve }],
    })),
  );
}

/** `true`: for this site. `"all"`: Download without asking anywhere; a Cancel isn't kept for every site. */
export type RememberDownload = boolean | "all";

/** Ignored unless it's the download being asked about. Remembered, it also settles the other waiting downloads it covers. */
export function answerDownload(request: DownloadRequest, allow: boolean, remember: RememberDownload): void {
  const { queue } = useApprovalStore.getState();
  if (queue[0] !== request) return;
  // Allowing a file that runs code is remembered for those too; allowing any other file isn't.
  const programs = allow && request.dangerous;
  const everywhere = remember === "all" && allow;
  const site = remember === true && request.origin !== "";
  if (site) useDownloadSitesStore.getState().setSite(request.origin, !allow ? "block" : programs ? "allowAll" : "allow");
  if (everywhere) {
    const prefs = useBrowserPrefsStore.getState();
    prefs.setAskBeforeDownloading(false);
    if (programs) prefs.setAskBeforeProgramDownloads(false);
  }
  const covered = (other: DownloadRequest) => !other.agent && !(allow && other.dangerous && !programs);
  const answered = everywhere
    ? queue.filter((other) => other === request || covered(other))
    : site
      ? queue.filter((other) => other === request || (other.origin === request.origin && covered(other)))
      : [request];
  useApprovalStore.setState({ queue: queue.filter((other) => !answered.includes(other)) });
  for (const other of answered) other.resolve(allow);
}
