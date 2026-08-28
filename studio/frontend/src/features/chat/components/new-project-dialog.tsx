// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useNavigate } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  type NativeProjectWorkspaceSelection,
  consumeNativePathToken,
  pickNativeProjectWorkspace,
  useNativePathLeasesSupported,
} from "@/features/native-intents";
import {
  ProjectSourceDropzone,
  type StagedSource,
  uploadStagedSources,
} from "@/features/rag/components/project-source-dropzone";
import { isTauri } from "@/lib/api-base";
import { toast } from "@/lib/toast";
import { Folder02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import { createChatProject } from "../hooks/use-chat-projects";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { ProjectRecord } from "../types";

function currentRoute(): string {
  if (typeof window === "undefined") return "";
  return window.location.pathname + window.location.search;
}

// Create-project dialog for the composer, sidebar, and projects page. Creating
// opens the new project; `onCreated` overrides that for callers with their own
// follow-up (the sidebar's "move this chat to a new project").
export function NewProjectDialog({
  open,
  onOpenChange,
  title = "Create project",
  submitLabel = "Create project",
  onCreated,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title?: string;
  submitLabel?: string;
  onCreated?: (
    project: ProjectRecord,
    context: { stayedOnRoute: boolean },
  ) => void | Promise<void>;
}) {
  const navigate = useNavigate();
  const nativePathLeasesSupported = useNativePathLeasesSupported();
  const [name, setName] = useState("");
  const [staged, setStaged] = useState<StagedSource[]>([]);
  const [busy, setBusy] = useState(false);
  const [pickingFolder, setPickingFolder] = useState(false);
  const [workspace, setWorkspace] =
    useState<NativeProjectWorkspaceSelection | null>(null);
  // A desktop drop reaches `staged` only once its native registration settles.
  // Creating before then would upload without the files the user just dropped.
  const [stagingDrop, setStagingDrop] = useState(false);
  // Uploads outlive this component, so a slow one must not yank the user to the
  // new project after they have navigated away.
  const mounted = useRef(true);
  useEffect(() => {
    // Set on setup, not just cleared on cleanup: StrictMode replays
    // setup/cleanup/setup, which would otherwise leave this false forever.
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  function reset() {
    setName("");
    setStaged([]);
    setStagingDrop(false);
    setWorkspace(null);
  }

  // Every close path routes through here: callers keep this mounted, so a draft
  // left behind would resurface (and upload) on the next project.
  function close() {
    if (busy) return;
    reset();
    onOpenChange(false);
  }

  async function commitCreate() {
    const trimmed = name.trim();
    if (!trimmed || busy || stagingDrop || pickingFolder) return;
    setBusy(true);
    // Sidebar callers keep this mounted across routes, so unmounting alone
    // cannot tell whether the user has moved on during a slow upload.
    const origin = currentRoute();
    // A folder grant is single-use, so a failed create cannot be retried with the
    // one already spent. Cleared on the way out either way, and said out loud on
    // failure: silently dropping it would make the next Create a managed project
    // without the user ever choosing one.
    let leaseSpent = false;
    try {
      const workspaceLease = workspace
        ? await consumeNativePathToken(
            workspace.token,
            "set-project-workspace",
          )
        : null;
      leaseSpent = Boolean(workspaceLease);
      const project = await createChatProject(trimmed, workspaceLease ? {
        nativePathLease: workspaceLease.nativePathLease,
      } : undefined);
      setWorkspace(null);
      // Upload before closing so the Sources panel lists them on first fetch.
      await uploadStagedSources(project.id, staged);
      if (!mounted.current) return;
      const stayedOnRoute = currentRoute() === origin;
      onOpenChange(false);
      reset();
      if (onCreated) {
        await onCreated(project, { stayedOnRoute });
        return;
      }
      if (!stayedOnRoute) return;
      const runtime = useChatRuntimeStore.getState();
      runtime.setActiveThreadId(null);
      runtime.setActiveProjectId(project.id);
      navigate({ to: "/chat", search: { project: project.id } });
    } catch (err) {
      const reason = err instanceof Error ? err.message : undefined;
      if (leaseSpent) setWorkspace(null);
      toast.error("Failed to create project", {
        description: leaseSpent
          ? `${reason ? `${reason} ` : ""}Choose the folder again to retry.`
          : reason,
      });
    } finally {
      setBusy(false);
    }
  }

  async function chooseWorkspace() {
    if (busy || pickingFolder) return;
    setPickingFolder(true);
    try {
      const selected = await pickNativeProjectWorkspace();
      if (selected) setWorkspace(selected);
    } catch (err) {
      toast.error("Couldn't open the folder picker", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setPickingFolder(false);
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (next) {
          onOpenChange(true);
          return;
        }
        close();
      }}
    >
      <DialogContent className="corner-squircle dialog-soft-surface min-w-0 gap-5 sm:max-w-xl">
        <DialogHeader>
          <DialogTitle className="text-ui-21">{title}</DialogTitle>
        </DialogHeader>
        {/* Name field: folder glyph in its own cell, divided from the input. */}
        <div className="flex items-stretch overflow-hidden rounded-[16px] border border-border bg-background transition-colors focus-within:border-ring has-[input:disabled]:opacity-50 dark:border-transparent dark:bg-white/[0.06]">
          <span className="flex w-9 shrink-0 items-center justify-center text-muted-foreground">
            <HugeiconsIcon
              icon={Folder02Icon}
              strokeWidth={1.75}
              className="size-5"
            />
          </span>
          <span aria-hidden="true" className="my-3 w-px bg-border" />
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void commitCreate();
              }
            }}
            autoFocus={true}
            disabled={busy}
            maxLength={120}
            placeholder="Project name"
            aria-label="Project name"
            className="min-w-0 flex-1 bg-transparent py-4 pr-4 pl-2.5 text-base outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed"
          />
        </div>
        {isTauri && nativePathLeasesSupported ? (
          <div className="space-y-2 rounded-[16px] border border-border bg-muted/20 p-3 dark:border-transparent dark:bg-white/[0.04]">
            <div className="flex min-w-0 flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="min-w-0">
                <p className="text-sm font-medium text-foreground">
                  Working directory
                </p>
                <p className="truncate text-xs text-muted-foreground">
                  {workspace?.path ?? "Unsloth managed folder"}
                </p>
              </div>
              <div className="flex flex-wrap gap-2 sm:shrink-0 sm:justify-end">
                {workspace ? (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    disabled={busy || pickingFolder}
                    onClick={() => setWorkspace(null)}
                  >
                    Use managed
                  </Button>
                ) : null}
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  disabled={busy || pickingFolder}
                  onClick={() => void chooseWorkspace()}
                >
                  {pickingFolder
                    ? "Choosing…"
                    : workspace
                      ? "Change folder"
                      : "Choose folder"}
                </Button>
              </div>
            </div>
          </div>
        ) : null}
        <ProjectSourceDropzone
          staged={staged}
          onChange={setStaged}
          disabled={busy}
          onPendingChange={setStagingDrop}
        />
        <DialogFooter className="flex-wrap gap-2 sm:justify-end">
          <Button type="button" variant="ghost" disabled={busy} onClick={close}>
            Cancel
          </Button>
          <Button
            type="button"
            onClick={() => void commitCreate()}
            disabled={!name.trim() || busy || stagingDrop || pickingFolder}
          >
            {busy ? "Creating…" : stagingDrop ? "Adding sources…" : submitLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
