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
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { useT } from "@/i18n";
import { useId, useState } from "react";
import { type RememberDownload, answerDownload, useApprovalStore } from "./download-approval-queue";

type Choice = "once" | "site" | "all";
const REMEMBER: Record<Choice, RememberDownload> = { once: false, site: true, all: "all" };

export function DownloadApprovalDialog() {
  const t = useT();
  const request = useApprovalStore((state) => state.queue[0]);
  const [choice, setChoice] = useState<Choice>("once");
  const id = useId();

  // Escape or an outside click cancels this once only, so a stray click with the box ticked doesn't block the site for good.
  const answer = (allow: boolean, kept = true) => {
    if (!request || useApprovalStore.getState().queue[0] !== request) return;
    setChoice("once");
    answerDownload(request, allow, kept ? REMEMBER[choice] : false);
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
        {request ? (
          <RadioGroup
            value={choice}
            onValueChange={(value) => setChoice(value as Choice)}
            aria-label={t("browser.downloadPrompt.remember")}
            className="gap-2"
          >
            {(["once", "site", "all"] as const)
              // Without a web origin (data:, an opaque blob) there's no site to remember.
              .filter((option) => option !== "site" || request.origin)
              .map((option) => (
                <div key={option} className="flex items-center gap-2">
                  <RadioGroupItem id={`${id}-${option}`} value={option} />
                  <Label htmlFor={`${id}-${option}`} className="cursor-pointer text-sm font-normal text-foreground">
                    {t(`browser.downloadPrompt.${option}`, { host: request.label })}
                  </Label>
                </div>
              ))}
          </RadioGroup>
        ) : null}
        <AlertDialogFooter>
          <AlertDialogCancel onClick={() => answer(false)}>{t("browser.downloadPrompt.cancel")}</AlertDialogCancel>
          <AlertDialogAction onClick={() => answer(true)}>{t("browser.downloadPrompt.download")}</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
