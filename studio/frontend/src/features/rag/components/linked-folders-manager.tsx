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
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Progress } from "@/components/ui/progress";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import { FolderAddIcon, FolderSyncIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { MoreHorizontalIcon, RotateCwIcon } from "lucide-react";
import { useState } from "react";
import type { FolderSyncJob, LinkedFolderScope } from "../types/rag";
import { useLinkedFolders } from "./use-linked-folders";

function percent(progress?: number | null): number | null {
  if (progress == null || !Number.isFinite(progress)) return null;
  return Math.max(0, Math.min(100, progress <= 1 ? progress * 100 : progress));
}

function jobSummary(job: FolderSyncJob): string {
  if (job.status === "failed") return job.error ?? "Sync failed";
  if (job.status === "completed") {
    const indexed = job.indexedFiles ?? job.processedFiles;
    return indexed == null
      ? "Sync complete"
      : `${indexed} file${indexed === 1 ? "" : "s"} indexed`;
  }
  const processed = job.processedFiles ?? 0;
  const discovered = job.discoveredFiles;
  return discovered == null
    ? job.stage || "Scanning folder"
    : `${processed} of ${discovered} files`;
}

export function LinkedFoldersManager({
  scope,
  compact = false,
  variant = "panel",
  onSourcesChanged,
}: {
  scope?: LinkedFolderScope;
  compact?: boolean;
  /** "panel" is the settings-style block with its own heading and Link folder button. "card" is
   *  the grouped list a dialog shows: one row per folder, an Add folder row at the foot of it. */
  variant?: "panel" | "card";
  onSourcesChanged?: () => void;
}) {
  const manager = useLinkedFolders(scope, onSourcesChanged);
  const [removeIndexFolder, setRemoveIndexFolder] = useState<{
    id: string;
    name: string;
  } | null>(null);

  /** Whether the folder has a sync in flight, which is what the row reports while it does. */
  function runningJob(folderId: string): FolderSyncJob | undefined {
    const job = manager.jobs[folderId];
    return job?.status === "pending" || job?.status === "running"
      ? job
      : undefined;
  }

  /** The row's own actions, identical in both layouts: a card row's "×" would have to drop three
   *  of them, and syncing a folder by hand is the reason most people open this list. */
  function folderMenu(folder: (typeof manager.folders)[number], running: boolean) {
    return (
      <DropdownMenu>
        <DropdownMenuTrigger asChild={true}>
          <Button
            type="button"
            size="icon-sm"
            variant="ghost"
            className="shrink-0 rounded-full"
            aria-label={`Actions for ${folder.displayName}`}
          >
            {running ? (
              <Spinner className="size-3.5" />
            ) : (
              <MoreHorizontalIcon className="size-4" />
            )}
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem
            disabled={running}
            onSelect={() => void manager.sync(folder.id)}
          >
            <HugeiconsIcon icon={FolderSyncIcon} strokeWidth={1.75} className="size-3.5" /> Sync changes
          </DropdownMenuItem>
          <DropdownMenuItem
            disabled={running}
            onSelect={() => void manager.rebuild(folder.id)}
          >
            <RotateCwIcon className="size-3.5" /> Rebuild index
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            onSelect={() => void manager.remove(folder.id, false)}
          >
            Unlink and keep indexed files
          </DropdownMenuItem>
          <DropdownMenuItem
            variant="destructive"
            onSelect={() =>
              setRemoveIndexFolder({
                id: folder.id,
                name: folder.displayName,
              })
            }
          >
            Unlink and remove indexed files
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    );
  }

  /** The confirmation the destructive unlink opens. Mounted by whichever layout rendered. */
  const removeIndexConfirm = (
    <AlertDialog
      open={removeIndexFolder !== null}
      onOpenChange={(open) => {
        if (!open) setRemoveIndexFolder(null);
      }}
    >
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Unlink folder and remove files?</AlertDialogTitle>
          <AlertDialogDescription>
            This will unlink &quot;{removeIndexFolder?.name}&quot; and remove all
            documents it manages from the index. The files on disk will not be
            changed.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            variant="destructive"
            onClick={() => {
              const folder = removeIndexFolder;
              setRemoveIndexFolder(null);
              if (folder) void manager.remove(folder.id, true);
            }}
          >
            Unlink and remove
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );

  // One grouped card: the folders, then the row that adds another. A dialog has a name field
  // above it in the same shape, so the two read as one stack rather than a panel inside a panel.
  if (variant === "card") {
    const rowClass =
      "flex min-w-0 items-center gap-3 px-3.5 py-2.5 border-t border-border/60 first:border-t-0 dark:border-[rgb(255_255_255_/_calc(0.08*var(--contrast-edge-gain,1)))]";
    return (
      <section className="flex min-w-0 flex-col gap-2">
        <div className="overflow-hidden rounded-[16px] border border-border bg-background dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))]">
          {manager.loading && manager.folders.length === 0 ? (
            <div className={cn(rowClass, "text-ui-13 text-muted-foreground")}>
              <Spinner className="size-4 shrink-0" />
              <span>Loading folders…</span>
            </div>
          ) : (
            manager.folders.map((folder) => {
              const job = manager.jobs[folder.id];
              const running = runningJob(folder.id) !== undefined;
              const failed =
                folder.status === "error" || job?.status === "failed";
              // A settled row is its name alone, the way a list of folders reads. The detail line
              // is for the states that need one; what is indexed stays on the row's tooltip.
              const detail = running
                ? jobSummary(job as FolderSyncJob)
                : failed
                  ? (job?.status === "failed" ? job.error : folder.error) ||
                    "Sync failed"
                  : null;
              return (
                <div
                  key={folder.id}
                  className={rowClass}
                  title={
                    folder.lastSyncedAt
                      ? `${folder.displayName}: last synced ${new Date(folder.lastSyncedAt).toLocaleString()}`
                      : `${folder.displayName}: ${folder.documentCount ?? 0} indexed documents`
                  }
                >
                  <HugeiconsIcon
                    icon={FolderSyncIcon}
                    strokeWidth={1.75}
                    className="size-4 shrink-0 text-muted-foreground"
                  />
                  <div className="min-w-0 flex-1">
                    <span className="block truncate text-sm">
                      {folder.displayName}
                    </span>
                    {detail ? (
                      <p
                        className={cn(
                          "truncate text-ui-11 text-muted-foreground",
                          failed && "text-destructive",
                        )}
                      >
                        {detail}
                      </p>
                    ) : null}
                    {running ? (
                      <Progress
                        value={percent(job?.progress) ?? 0}
                        aria-label={`Sync progress for ${folder.displayName}`}
                        className="mt-1.5 h-1"
                      />
                    ) : null}
                  </div>
                  {folderMenu(folder, running)}
                </div>
              );
            })
          )}
          {scope ? (
            <button
              type="button"
              disabled={!manager.desktopSupported || manager.mutating}
              onClick={() => void manager.link()}
              // Named for what it does; the title is the tooltip saying why it cannot, which
              // would otherwise be read out as the row's name.
              aria-label="Add folder"
              title={
                manager.desktopSupported
                  ? "Choose a local folder"
                  : "Requires the managed desktop backend"
              }
              className={cn(
                rowClass,
                "w-full cursor-pointer py-3 text-left text-sm transition-colors hover:bg-muted/60 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent",
              )}
            >
              {manager.mutating ? (
                <Spinner className="size-4 shrink-0" />
              ) : (
                <HugeiconsIcon
                  icon={FolderAddIcon}
                  strokeWidth={1.75}
                  className="size-4 shrink-0 text-muted-foreground"
                />
              )}
              <span>Add folder</span>
            </button>
          ) : null}
        </div>
        {/* Only where it changes what the row above can do: in the browser build there is no
            native folder picker, and the rows already linked keep syncing regardless. */}
        {!manager.desktopSupported ? (
          <p className="px-1 text-ui-11 text-muted-foreground">
            Linking a folder needs the managed desktop backend. Folders already
            linked stay synced.
          </p>
        ) : null}
        {removeIndexConfirm}
      </section>
    );
  }

  return (
    <section className={cn("flex min-w-0 flex-col gap-3", compact && "gap-2")}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-sm font-medium text-foreground">
            Linked local folders
          </h3>
          <p className="text-xs leading-snug text-muted-foreground">
            {manager.desktopSupported
              ? "Keep supported documents indexed as this folder changes."
              : "Existing linked folders stay synced; linking requires the managed desktop backend."}
          </p>
        </div>
        {scope ? (
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="shrink-0"
            disabled={!manager.desktopSupported || manager.mutating}
            onClick={() => void manager.link()}
            title={
              manager.desktopSupported
                ? "Choose a local folder"
                : "Requires the managed desktop backend"
            }
          >
            {manager.mutating ? (
              <Spinner className="size-3.5" />
            ) : (
              <HugeiconsIcon icon={FolderSyncIcon} strokeWidth={1.75} className="size-3.5" />
            )}
            Link folder
          </Button>
        ) : null}
      </div>

      {manager.loading && manager.folders.length === 0 ? (
        <div className="flex justify-center py-4">
          <Spinner />
        </div>
      ) : manager.folders.length === 0 ? (
        <div className="rounded-xl border border-dashed px-4 py-5 text-center text-xs text-muted-foreground">
          No linked folders.
        </div>
      ) : (
        <ul className="flex flex-col gap-1.5">
          {manager.folders.map((folder) => {
            const job = manager.jobs[folder.id];
            const running =
              job?.status === "pending" || job?.status === "running";
            const progress = percent(job?.progress);
            return (
              <li
                key={folder.id}
                className="flex min-w-0 items-start gap-3 rounded-xl border border-border/70 bg-background px-3 py-2.5"
              >
                <HugeiconsIcon icon={FolderSyncIcon} strokeWidth={1.75} className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline gap-2">
                    <span
                      className="truncate text-sm font-medium"
                      title={folder.displayName}
                    >
                      {folder.displayName}
                    </span>
                    {scope ? null : (
                      <span
                        className="max-w-48 shrink truncate text-ui-11 text-muted-foreground"
                        title={
                          folder.scopeName ||
                          `${folder.scopeType === "knowledge_base" ? "Knowledge base" : "Project"} ${folder.scopeId}`
                        }
                      >
                        {folder.scopeName ||
                          `${folder.scopeType === "knowledge_base" ? "Knowledge base" : "Project"} ${folder.scopeId}`}
                      </span>
                    )}
                  </div>
                  <p
                    className={cn(
                      "text-ui-11 text-muted-foreground",
                      (folder.status === "error" || job?.status === "failed") &&
                        "text-destructive",
                    )}
                  >
                    {job
                      ? jobSummary(job)
                      : folder.error ||
                        (folder.lastSyncedAt
                          ? `Last synced ${new Date(folder.lastSyncedAt).toLocaleString()}`
                          : `${folder.documentCount ?? 0} indexed documents`)}
                  </p>
                  {running ? (
                    <Progress
                      value={progress ?? 0}
                      aria-label={`Sync progress for ${folder.displayName}`}
                      className="mt-2 h-1.5"
                    />
                  ) : null}
                </div>
                {folderMenu(folder, running)}
              </li>
            );
          })}
        </ul>
      )}
      {removeIndexConfirm}
    </section>
  );
}
