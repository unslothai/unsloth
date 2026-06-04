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
  AlertDialogMedia,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { openLink } from "@/lib/open-link";
import { LinkSquare02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useExternalLinkConfirm } from "../stores/external-link-confirm";

function hostOf(url: string): string {
  try {
    return new URL(url, "https://_").host;
  } catch {
    return url;
  }
}

export function ExternalLinkConfirmDialog() {
  const pendingUrl = useExternalLinkConfirm((s) => s.pendingUrl);
  const dismiss = useExternalLinkConfirm((s) => s.dismiss);
  const open = pendingUrl !== null;

  return (
    <AlertDialog
      open={open}
      onOpenChange={(next) => {
        if (!next) dismiss();
      }}
    >
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogMedia>
            <HugeiconsIcon
              icon={LinkSquare02Icon}
              strokeWidth={1.75}
              className="text-muted-foreground"
            />
          </AlertDialogMedia>
          <AlertDialogTitle>Open external link</AlertDialogTitle>
          <AlertDialogDescription>
            You are about to leave Unsloth and open this page in your browser.
            Only continue if you trust the destination.
          </AlertDialogDescription>
        </AlertDialogHeader>
        {pendingUrl && (
          <div className="min-w-0 rounded-[12px] bg-muted/50 px-3 py-2.5 text-left">
            <p className="truncate text-[13px] font-medium text-foreground">
              {hostOf(pendingUrl)}
            </p>
            <p className="mt-0.5 break-all text-[11.5px] leading-[16px] text-muted-foreground">
              {pendingUrl}
            </p>
          </div>
        )}
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            onClick={() => {
              if (pendingUrl) openLink(pendingUrl);
              dismiss();
            }}
          >
            Open link
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
