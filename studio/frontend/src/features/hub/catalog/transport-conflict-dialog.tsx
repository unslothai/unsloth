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
import type { TransportConflictInfo } from "@/features/hub/download-manager";

export type { TransportConflictInfo } from "@/features/hub/download-manager";

export function TransportConflictDialog({
  conflict,
  onCancel,
  onKeepTransport,
  onSwitchTransport,
}: {
  conflict: TransportConflictInfo | null;
  onCancel: () => void;
  // Continue on the partial download's existing transport (resumes if possible);
  // distinct from switching transport, which always restarts from scratch.
  onKeepTransport: () => void;
  onSwitchTransport: () => void;
}) {
  const previousLabel = conflict?.previous.toUpperCase() ?? "";
  const nextLabel = conflict?.next.toUpperCase() ?? "";
  const description = conflict?.resumable ? (
    <>
      You started this download with{" "}
      <span className="font-medium text-foreground">{previousLabel}</span>, but
      your current setting is{" "}
      <span className="font-medium text-foreground">{nextLabel}</span>. Resume
      with {previousLabel} to keep the progress you already have, or restart
      with {nextLabel} to begin from scratch.
    </>
  ) : (
    <>
      Your previous download used{" "}
      <span className="font-medium text-foreground">{previousLabel}</span>,
      which can't resume its partial files, so the existing partial will be
      discarded and this download will start from the beginning either way.
      Restart with {previousLabel} to keep it usually faster, or restart with{" "}
      {nextLabel} so future cancels can resume.
    </>
  );
  const primaryLabel = conflict?.resumable
    ? `Resume with ${previousLabel}`
    : `Restart with ${previousLabel}`;
  const secondaryLabel = `Restart with ${nextLabel}`;
  return (
    <AlertDialog
      open={conflict !== null}
      onOpenChange={(o) => {
        if (!o) onCancel();
      }}
    >
      <AlertDialogContent className="sm:!max-w-[352px]">
        <AlertDialogHeader>
          <AlertDialogTitle>Different transport mode</AlertDialogTitle>
          <AlertDialogDescription>{description}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter className="!flex !flex-col gap-2 sm:!flex-col sm:!justify-stretch">
          <AlertDialogAction
            className="w-full"
            onClick={(e) => {
              e.preventDefault();
              onKeepTransport();
            }}
          >
            {primaryLabel}
          </AlertDialogAction>
          <AlertDialogAction
            variant="outline"
            className="w-full !bg-transparent hover:!bg-transparent"
            onClick={(e) => {
              e.preventDefault();
              onSwitchTransport();
            }}
          >
            {secondaryLabel}
          </AlertDialogAction>
          <AlertDialogCancel className="w-full !bg-transparent hover:!bg-transparent">
            Cancel
          </AlertDialogCancel>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
