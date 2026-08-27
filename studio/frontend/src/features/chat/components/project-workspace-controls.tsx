// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState } from "react";

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
  pickNativeProjectFolder,
  useNativePathLeasesSupported,
} from "@/features/native-intents";
import { toast } from "@/lib/toast";

import { projectWorkspaceMutationShouldStayBusy } from "../api/project-workspace-mutation";
import {
  changeChatProjectWorkspaceFolder,
  disconnectChatProjectWorkspaceFolder,
} from "../hooks/use-chat-projects";
import type { ProjectRecord } from "../types";

export function ProjectWorkspaceControls({
  project,
}: {
  project: ProjectRecord;
}) {
  return (
    <ProjectWorkspaceControlsInner
      key={`${project.id}:${project.workspaceRevision || 0}`}
      project={project}
    />
  );
}

function ProjectWorkspaceControlsInner({
  project,
}: {
  project: ProjectRecord;
}) {
  const nativePathLeasesSupported = useNativePathLeasesSupported();
  const [busy, setBusy] = useState(false);
  const [confirmDisconnect, setConfirmDisconnect] = useState<{
    projectId: string;
    workspaceRevision: number;
  } | null>(null);
  const mounted = useRef(false);
  const target = {
    projectId: project.id,
    workspaceRevision: project.workspaceRevision || 0,
  };
  const folderBacked = project.workspaceKind === "folder";
  const unavailable = folderBacked && project.workspaceAvailable === false;

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  async function chooseFolder(): Promise<void> {
    if (busy || !nativePathLeasesSupported) return;
    setBusy(true);
    let keepBusy = false;
    try {
      const selected = await pickNativeProjectFolder();
      if (!selected) return;
      if (!mounted.current) {
        return;
      }
      await changeChatProjectWorkspaceFolder(
        target.projectId,
        selected.token,
        target.workspaceRevision,
      );
    } catch (error) {
      keepBusy = projectWorkspaceMutationShouldStayBusy(error);
      toast.error(
        unavailable ? "Could not reconnect folder" : "Could not change folder",
        { description: error instanceof Error ? error.message : undefined },
      );
    } finally {
      if (mounted.current && !keepBusy) {
        setBusy(false);
      }
    }
  }

  async function disconnect(): Promise<void> {
    const target = confirmDisconnect;
    setConfirmDisconnect(null);
    if (!target) {
      return;
    }
    setBusy(true);
    let keepBusy = false;
    try {
      await disconnectChatProjectWorkspaceFolder(
        target.projectId,
        target.workspaceRevision,
      );
    } catch (error) {
      keepBusy = projectWorkspaceMutationShouldStayBusy(error);
      toast.error("Could not disconnect folder", {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      if (mounted.current && !keepBusy) {
        setBusy(false);
      }
    }
  }

  return (
    <>
      <section
        className={`mb-4 rounded-[18px] border px-4 py-3 ${
          unavailable
            ? "border-destructive/25 bg-destructive/5"
            : "border-border bg-muted/25"
        }`}
        aria-live="polite"
      >
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <p className="text-sm font-semibold text-foreground">
              {unavailable
                ? "Local project folder unavailable"
                : folderBacked
                  ? "Existing folder"
                  : "Unsloth-managed workspace"}
            </p>
            {folderBacked ? (
              <code className="mt-1 block break-all text-xs text-muted-foreground">
                {project.workspacePath || "Folder path unavailable"}
              </code>
            ) : (
              <p className="mt-1 text-xs text-muted-foreground">
                Chats and tools use private storage managed by Unsloth.
              </p>
            )}
            {unavailable ? (
              <p className="mt-1 text-xs text-muted-foreground">
                Choose the folder again, or disconnect it to return to the
                managed workspace. The existing folder remains untouched.
              </p>
            ) : null}
          </div>
          <div className="flex shrink-0 flex-wrap gap-2">
            <Button
              type="button"
              size="sm"
              variant="outline"
              disabled={busy || !nativePathLeasesSupported}
              onClick={() => void chooseFolder()}
            >
              {busy
                ? "Updating..."
                : unavailable
                  ? "Choose folder"
                  : folderBacked
                    ? "Change folder"
                    : "Use existing folder"}
            </Button>
            {folderBacked ? (
              <Button
                type="button"
                size="sm"
                variant="ghost"
                disabled={busy}
                onClick={() => setConfirmDisconnect({ ...target })}
              >
                Disconnect
              </Button>
            ) : null}
          </div>
        </div>
      </section>

      <AlertDialog
        open={confirmDisconnect !== null}
        onOpenChange={(open) => {
          if (!open) setConfirmDisconnect(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Disconnect existing folder?</AlertDialogTitle>
            <AlertDialogDescription>
              This project will return to its Unsloth-managed workspace. The
              existing folder and every file in it will remain untouched.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={() => void disconnect()}>
              Disconnect
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
