// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Checkbox } from "@/components/ui/checkbox";
import { getLocale, translate, useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useId, useState } from "react";
import { create } from "zustand";
import { hostOf, isWebUrl } from "./address";
import { useBrowserPrefsStore } from "./prefs-store";

/** `host` is "" when there is no site to remember the answer for. */
type Request = { host: string; label: string; name: string; resolve: (allow: boolean) => void };

const useApprovalStore = create<{ queue: Request[] }>(() => ({ queue: [] }));

/** Whether a file from `url` may be saved: a remembered answer for its site, else the user's.
 *  `site` is the page that started it, which the answer is kept for (the file's own address by
 *  default). Only a web page counts as a site: blob: and data: URLs have no host, and one answer
 *  for "" would then cover them on every site. */
export function approveDownload(url: string, name: string, site: string = url): Promise<boolean> {
  const host = isWebUrl(site) ? hostOf(site) : "";
  const prefs = useBrowserPrefsStore.getState();
  const remembered = host ? prefs.downloadSites[host] : undefined;
  if (remembered === "block") {
    toast.error(translate("browser.downloadPrompt.blocked", { host }, getLocale()));
    return Promise.resolve(false);
  }
  if (remembered === "allow" || !prefs.askBeforeDownloading) return Promise.resolve(true);
  const label = host || hostOf(url) || url.slice(0, 80);
  return new Promise((resolve) =>
    useApprovalStore.setState((state) => ({ queue: [...state.queue, { host, label, name, resolve }] })),
  );
}

/** Asks about each waiting download in turn; mounted once for the app. */
export function DownloadApprovalDialog() {
  const t = useT();
  const request = useApprovalStore((state) => state.queue[0]);
  const [remember, setRemember] = useState(false);
  const checkboxId = useId();

  // `kept`: a button press. Escape or a click outside only cancels this once, so a stray click
  // with the box ticked doesn't block the site for good.
  const answer = (allow: boolean, kept = true) => {
    // A button press also closes the dialog; only the first answer counts for this request.
    if (!request || useApprovalStore.getState().queue[0] !== request) return;
    if (kept && remember && request.host) {
      useBrowserPrefsStore.getState().setDownloadSite(request.host, allow ? "allow" : "block");
    }
    setRemember(false);
    useApprovalStore.setState((state) => ({ queue: state.queue.slice(1) }));
    request.resolve(allow);
  };

  return (
    <AlertDialog open={request !== undefined} onOpenChange={(open) => !open && answer(false, false)}>
      <AlertDialogContent onOverlayClick={() => answer(false, false)}>
        <AlertDialogHeader>
          <AlertDialogTitle>{t("browser.downloadPrompt.title")}</AlertDialogTitle>
          <AlertDialogDescription className="break-words">
            {request ? t("browser.downloadPrompt.description", { host: request.label, name: request.name }) : null}
          </AlertDialogDescription>
        </AlertDialogHeader>
        {request?.host ? (
          <label htmlFor={checkboxId} className="flex cursor-pointer items-center gap-2 text-sm text-foreground">
            <Checkbox id={checkboxId} checked={remember} onCheckedChange={(checked) => setRemember(checked === true)} />
            {t("browser.downloadPrompt.remember")}
          </label>
        ) : null}
        <AlertDialogFooter>
          <AlertDialogCancel onClick={() => answer(false)}>{t("browser.downloadPrompt.cancel")}</AlertDialogCancel>
          <AlertDialogAction onClick={() => answer(true)}>{t("browser.downloadPrompt.download")}</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
