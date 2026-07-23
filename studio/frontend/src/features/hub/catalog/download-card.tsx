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
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  ArrowReloadHorizontalIcon,
  Delete02Icon,
  Download01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  DownloadProgressBar,
  type DownloadJob,
  type DownloadJobProgress,
} from "../download-manager";
import { DownloadCancelIndicator } from "./download-cancel-indicator";
import { TransportConflictDialog } from "./transport-conflict-dialog";
import {
  downloadActionAriaLabel,
  downloadActionLabel,
} from "./use-download-card-state";

/**
 * Shared shell for every download surface (safetensors, GGUF, dataset): card
 * frame, progress bar, transport-conflict dialog, plus card-specific `dialogs`
 * and children.
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
      <div className="hub-download-card">
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

export function CardDeleteButton({
  label,
  onClick,
}: {
  label: string;
  onClick: () => void;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={label}
          onClick={(e) => {
            e.stopPropagation();
            onClick();
          }}
          className="inline-flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground opacity-0 transition-[opacity,background-color,color] duration-150 hover:bg-rose-500/10 hover:text-rose-600 focus-visible:opacity-100 group-hover/dl:opacity-100 dark:hover:bg-rose-500/15 dark:hover:text-rose-400"
        >
          <HugeiconsIcon
            icon={Delete02Icon}
            strokeWidth={1.75}
            className="size-4"
          />
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" sideOffset={4}>
        Delete from device
      </TooltipContent>
    </Tooltip>
  );
}

export function CardUpdateButton({
  label,
  onClick,
  emphasized = false,
}: {
  label: string;
  onClick: () => void;
  /** When true (a newer revision is available) the control becomes a prominent
   *  labeled amber pill instead of the quiet hover-revealed icon — the
   *  "update available" cue. Amber is the established status tone in this surface
   *  (the older-cache hint banner), kept tinted (never solid) per the design
   *  system's feedback-color convention, so the one emerald accent stays scarce. */
  emphasized?: boolean;
}) {
  if (emphasized) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            aria-label={label}
            onClick={(e) => {
              e.stopPropagation();
              onClick();
            }}
            className="inline-flex h-7 shrink-0 cursor-pointer items-center gap-1.5 rounded-full bg-amber-500/[0.07] pl-2 pr-2.5 text-[0.75rem] font-medium text-amber-800/90 transition-colors duration-150 hover:bg-amber-500/[0.12] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-amber-500/25 dark:bg-amber-400/[0.08] dark:text-amber-200/85 dark:hover:bg-amber-400/[0.16]"
          >
            <HugeiconsIcon
              icon={ArrowReloadHorizontalIcon}
              strokeWidth={2}
              className="size-3.5"
            />
            Update
          </button>
        </TooltipTrigger>
        <TooltipContent side="top" sideOffset={4}>
          A newer version is available on Hugging Face
        </TooltipContent>
      </Tooltip>
    );
  }
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={label}
          onClick={(e) => {
            e.stopPropagation();
            onClick();
          }}
          className="inline-flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground opacity-0 transition-[opacity,background-color,color] duration-150 hover:bg-amber-500/10 hover:text-amber-600 focus-visible:opacity-100 group-hover/dl:opacity-100 dark:hover:bg-amber-500/15 dark:hover:text-amber-400"
        >
          <HugeiconsIcon
            icon={ArrowReloadHorizontalIcon}
            strokeWidth={1.75}
            className="size-4"
          />
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" sideOffset={4}>
        Update from Hugging Face
      </TooltipContent>
    </Tooltip>
  );
}

/** Download / Cancel / Resume button for the safetensors and dataset cards. */
export function DownloadActionButton({
  downloading,
  cancelling,
  loading = false,
  isPartial = false,
  partialTransport = null,
  progressPercent = null,
  disabled,
  onClick,
  className,
}: {
  downloading: boolean;
  cancelling: boolean;
  loading?: boolean;
  isPartial?: boolean;
  partialTransport?: string | null;
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
      aria-label={downloadActionAriaLabel(downloading, cancelling)}
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
          {downloadActionLabel(isPartial, partialTransport)}
        </>
      )}
    </button>
  );
}

/** Confirmation dialog shared by the model, quantization, and dataset delete flows. */
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

/** Confirmation dialog shared by the model and quantization update flows. */
export function UpdateConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  updating,
  onConfirm,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: ReactNode;
  updating: boolean;
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
          <AlertDialogCancel disabled={updating}>Cancel</AlertDialogCancel>
          <AlertDialogAction
            variant="default"
            disabled={updating}
            onClick={(e) => {
              e.preventDefault();
              onConfirm();
            }}
          >
            {updating ? "Updating…" : "Update"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
