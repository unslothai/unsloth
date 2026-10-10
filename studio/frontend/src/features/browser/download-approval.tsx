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
import { useT } from "@/i18n";
import { useId, useState } from "react";
import { answerDownload, useApprovalStore } from "./download-approval-queue";

export function DownloadApprovalDialog() {
  const t = useT();
  const request = useApprovalStore((state) => state.queue[0]);
  const [remember, setRemember] = useState(false);
  const checkboxId = useId();

  // Escape or an outside click cancels this once only, so a stray click with the box ticked doesn't block the site for good.
  const answer = (allow: boolean, kept = true) => {
    if (!request || useApprovalStore.getState().queue[0] !== request) return;
    setRemember(false);
    answerDownload(request, allow, kept && remember);
  };

  return (
    <AlertDialog open={request !== undefined} onOpenChange={(open) => !open && answer(false, false)}>
      <AlertDialogContent onOverlayClick={() => answer(false, false)}>
        <AlertDialogHeader>
          <AlertDialogTitle>{t("browser.downloadPrompt.title")}</AlertDialogTitle>
          <AlertDialogDescription className="break-words">
            {request ? t("browser.downloadPrompt.description", { host: request.label, name: request.name }) : null}
          </AlertDialogDescription>
          {request?.dangerous ? (
            <p className="text-sm font-medium text-destructive">
              {t("browser.downloadPrompt.dangerous", { host: request.label })}
            </p>
          ) : null}
        </AlertDialogHeader>
        {/* Files that run code ask every time, so there's nothing to remember. */}
        {request?.origin && !request.dangerous ? (
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
