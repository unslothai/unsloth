// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactNode } from "react";
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
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import { Download01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { DownloadJob, DownloadJobProgress } from "../hooks/use-download-job";
import { DownloadCancelIndicator } from "./download-cancel-indicator";
import { DownloadProgressBar } from "./download-progress-bar";
import { TransportConflictDialog } from "./transport-conflict-dialog";

/**
 * Shared shell for every download surface (safetensors, GGUF, dataset): the
 * card frame, the progress bar, the transport-conflict dialog, and any
 * card-specific dialogs passed via `dialogs`. Each card supplies its own row
 * content (info/selector + actions) as children.
 */
export function DownloadCard({
  job,
  progress,
  children,
  dialogs,
}: {
  job: DownloadJob;
  progress: DownloadJobProgress | null;
  children: ReactNode;
  dialogs?: ReactNode;
}) {
  return (
    <>
      <div className="download-card">
        <div className="group/dl flex items-center">{children}</div>
        {progress && (
          <DownloadProgressBar progress={progress} bytesPerSec={job.bytesPerSec} />
        )}
      </div>
      <TransportConflictDialog
        conflict={job.transportConflict}
        onCancel={job.cancelConflict}
        onKeepTransport={job.resumeConflict}
        onSwitchTransport={job.restartConflict}
      />
      {dialogs}
    </>
  );
}

/** Vertical hairline that fades out on row hover, separating info from actions. */
export function CardDivider() {
  return (
    <div
      aria-hidden="true"
      className="ml-1 mr-0 h-5 w-px shrink-0 bg-foreground/[0.06] opacity-100 transition-opacity duration-150 group-hover/dl:opacity-0 dark:bg-white/[0.04]"
    />
  );
}

/**
 * Download / Cancel / Resume button shared by the safetensors and dataset
 * cards. The GGUF card folds Run/Chat into its CTA and stays bespoke.
 */
export function DownloadActionButton({
  downloading,
  cancelling,
  loading = false,
  isPartial = false,
  progressPercent = null,
  disabled,
  onClick,
  className,
}: {
  downloading: boolean;
  cancelling: boolean;
  loading?: boolean;
  isPartial?: boolean;
  progressPercent?: number | null;
  disabled: boolean;
  onClick: () => void;
  className?: string;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      aria-label={
        cancelling ? "Cancelling…" : downloading ? "Cancel download" : undefined
      }
      className={cn(
        "hub-action-btn w-28",
        (loading || cancelling) && "opacity-70",
        downloading &&
          !cancelling &&
          "hover:bg-rose-500/10 hover:text-rose-600 dark:hover:text-rose-400",
        className,
      )}
    >
      {cancelling ? (
        <span className="inline-flex items-center gap-2 text-muted-foreground">
          <Spinner />
          Cancelling…
        </span>
      ) : downloading ? (
        <>
          <DownloadCancelIndicator />
          {progressPercent != null ? `${progressPercent}%` : null}
        </>
      ) : loading ? (
        <>
          <Spinner />
          Loading…
        </>
      ) : (
        <>
          <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} />
          {isPartial ? "Resume" : "Download"}
        </>
      )}
    </button>
  );
}

/**
 * Confirmation dialog shared by the model, quantization, and dataset delete
 * flows. Callers own the open/deleting state and supply the body copy.
 */
export function DeleteConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  deleting,
  onConfirm,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: ReactNode;
  deleting: boolean;
  onConfirm: () => void;
}) {
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent size="sm">
        <AlertDialogHeader>
          <AlertDialogTitle>{title}</AlertDialogTitle>
          <AlertDialogDescription>{description}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={deleting}>Cancel</AlertDialogCancel>
          <AlertDialogAction
            variant="destructive"
            disabled={deleting}
            onClick={(e) => {
              e.preventDefault();
              onConfirm();
            }}
          >
            {deleting ? "Deleting…" : "Delete"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
