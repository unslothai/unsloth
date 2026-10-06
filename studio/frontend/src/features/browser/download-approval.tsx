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
import { hostOf } from "./address";
import { useBrowserPrefsStore } from "./prefs-store";

type Request = { host: string; name: string; resolve: (allow: boolean) => void };

const useApprovalStore = create<{ queue: Request[] }>(() => ({ queue: [] }));

/** Whether a file from `url` may be saved: a remembered answer for its site, else the user's. */
export function approveDownload(url: string, name: string): Promise<boolean> {
  const host = hostOf(url);
  const prefs = useBrowserPrefsStore.getState();
  const remembered = prefs.downloadSites[host];
  if (remembered === "block") {
    toast.error(translate("browser.downloadPrompt.blocked", { host }, getLocale()));
    return Promise.resolve(false);
  }
  if (remembered === "allow" || !prefs.askBeforeDownloading) return Promise.resolve(true);
  return new Promise((resolve) =>
    useApprovalStore.setState((state) => ({ queue: [...state.queue, { host, name, resolve }] })),
  );
}

/** Asks about each waiting download in turn; mounted once for the app. */
export function DownloadApprovalDialog() {
  const t = useT();
  const request = useApprovalStore((state) => state.queue[0]);
  const [remember, setRemember] = useState(false);
  const checkboxId = useId();

  const answer = (allow: boolean) => {
    // A button press also closes the dialog; only the first answer counts for this request.
    if (!request || useApprovalStore.getState().queue[0] !== request) return;
    if (remember) useBrowserPrefsStore.getState().setDownloadSite(request.host, allow ? "allow" : "block");
    setRemember(false);
    useApprovalStore.setState((state) => ({ queue: state.queue.slice(1) }));
    request.resolve(allow);
  };

  return (
    <AlertDialog open={request !== undefined} onOpenChange={(open) => !open && answer(false)}>
      <AlertDialogContent onOverlayClick={() => answer(false)}>
        <AlertDialogHeader>
          <AlertDialogTitle>{t("browser.downloadPrompt.title")}</AlertDialogTitle>
          <AlertDialogDescription className="break-words">
            {request ? t("browser.downloadPrompt.description", { host: request.host, name: request.name }) : null}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <label htmlFor={checkboxId} className="flex cursor-pointer items-center gap-2 text-sm text-foreground">
          <Checkbox id={checkboxId} checked={remember} onCheckedChange={(checked) => setRemember(checked === true)} />
          {t("browser.downloadPrompt.remember")}
        </label>
        <AlertDialogFooter>
          <AlertDialogCancel onClick={() => answer(false)}>{t("browser.downloadPrompt.cancel")}</AlertDialogCancel>
          <AlertDialogAction onClick={() => answer(true)}>{t("browser.downloadPrompt.download")}</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
