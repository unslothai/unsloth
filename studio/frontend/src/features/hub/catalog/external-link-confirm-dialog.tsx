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
      <AlertDialogContent className="gap-5 sm:max-w-[calc(400px*var(--ui-space-scale,1))]">
        <AlertDialogHeader className="place-items-stretch gap-2 text-left">
          <div className="flex items-center gap-3">
            <span className="inline-flex size-9 shrink-0 items-center justify-center rounded-full bg-muted">
              <HugeiconsIcon
                icon={LinkSquare02Icon}
                strokeWidth={1.75}
                className="size-4.5 text-muted-foreground"
              />
            </span>
            <AlertDialogTitle>Open external link</AlertDialogTitle>
          </div>
          <AlertDialogDescription>
            You are about to leave Unsloth and open this page in your browser.
            Only continue if you trust the destination.
          </AlertDialogDescription>
        </AlertDialogHeader>
        {pendingUrl && (
          <div className="min-w-0 rounded-[12px] bg-muted/50 px-3.5 py-3 text-left">
            <p className="truncate text-ui-13 font-medium text-foreground">
              {hostOf(pendingUrl)}
            </p>
            <p className="mt-0.5 break-all text-ui-11p5 leading-ui-16 text-muted-foreground">
              {pendingUrl}
            </p>
          </div>
        )}
        <AlertDialogFooter className="grid grid-cols-2">
          <AlertDialogCancel className="w-full">Cancel</AlertDialogCancel>
          <AlertDialogAction
            className="w-full"
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
