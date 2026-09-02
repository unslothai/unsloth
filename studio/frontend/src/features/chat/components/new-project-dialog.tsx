// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useNavigate } from "@tanstack/react-router";
import {
  useEffect,
  useId,
  useMemo,
  useState,
  useSyncExternalStore,
} from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  pickNativeProjectFolder,
  useNativePathLeasesSupported,
} from "@/features/native-intents";
// This dialog deliberately imports the small dropzone module directly. The
// RAG barrel also evaluates page-level UI that is unrelated to project setup.
// eslint-disable-next-line no-restricted-imports
import {
  ProjectSourceDropzone,
  type StagedSource,
  uploadStagedSources,
} from "@/features/rag/components/project-source-dropzone";
import { isTauri } from "@/lib/api-base";
import { toast } from "@/lib/toast";
import { Folder02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import {
  createChatProject,
  openChatProjectFromFolder,
} from "../hooks/use-chat-projects";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { ProjectRecord } from "../types";
import {
  type NewProjectDialogDependencies,
  createNewProjectDialogController,
} from "./new-project-dialog-controller";

function currentRoute(): string {
  if (typeof window === "undefined") {
    return "";
  }
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
  const folderSupportDescriptionId = useId();
  const nativePathLeasesSupported = useNativePathLeasesSupported();
  const dependencies = useMemo<NewProjectDialogDependencies<StagedSource>>(
    () => ({
      nativePathLeasesSupported,
      pickFolder: pickNativeProjectFolder,
      createManaged: createChatProject,
      openFolder: openChatProjectFromFolder,
      uploadSources: uploadStagedSources,
      currentRoute,
      onOpenChange,
      onCreated,
      activateProject(projectId) {
        const runtime = useChatRuntimeStore.getState();
        runtime.setActiveThreadId(null);
        runtime.setActiveProjectId(projectId);
      },
      navigateToProject(projectId) {
        navigate({ to: "/chat", search: { project: projectId } });
      },
      showError(message, description) {
        toast.error(message, { description });
      },
    }),
    [nativePathLeasesSupported, navigate, onCreated, onOpenChange],
  );
  const [controller] = useState(() =>
    createNewProjectDialogController(dependencies),
  );
  const dialogState = useSyncExternalStore(
    controller.subscribe,
    controller.getState,
    controller.getState,
  );
  const {
    name,
    sources: staged,
    workspaceMode,
    folder,
    pickingFolder,
    busy,
    stagingSources: stagingDrop,
  } = dialogState;

  useEffect(() => {
    controller.updateDependencies(dependencies);
  }, [controller, dependencies]);

  useEffect(() => {
    controller.mount();
    return controller.unmount;
  }, [controller]);

  return (
    <Dialog open={open} onOpenChange={controller.openChanged}>
      <DialogContent className="corner-squircle dialog-soft-surface gap-5 sm:max-w-lg">
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
            onChange={(e) => controller.setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void controller.submit();
              }
            }}
            disabled={busy}
            maxLength={120}
            placeholder="Project name"
            aria-label="Project name"
            className="min-w-0 flex-1 bg-transparent py-4 pr-4 pl-2.5 text-base outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed"
          />
        </div>
        <section className="space-y-2">
          <div>
            <p className="text-sm font-medium text-foreground">Workspace</p>
            <p className="text-xs text-muted-foreground">
              Choose an Unsloth-managed workspace or work directly in an
              existing local project folder.
            </p>
          </div>
          <fieldset
            className="grid grid-cols-1 gap-2 sm:grid-cols-2"
            aria-describedby={folderSupportDescriptionId}
          >
            <legend className="sr-only">Project workspace</legend>
            <Button
              type="button"
              variant={workspaceMode === "managed" ? "default" : "outline"}
              disabled={busy}
              aria-pressed={workspaceMode === "managed"}
              onClick={controller.clickManagedWorkspace}
            >
              Managed workspace
            </Button>
            <Button
              type="button"
              variant={workspaceMode === "folder" ? "default" : "outline"}
              disabled={controller.folderPickerDisabled()}
              aria-pressed={workspaceMode === "folder"}
              aria-describedby={folderSupportDescriptionId}
              onClick={() => void controller.clickExistingFolder()}
              title={
                nativePathLeasesSupported
                  ? "Choose an existing local folder"
                  : isTauri
                    ? "Existing folders require the managed desktop backend"
                    : "Existing folders require the desktop app"
              }
            >
              {pickingFolder
                ? "Choosing…"
                : folder
                  ? folder.displayName
                  : "Use existing folder"}
            </Button>
          </fieldset>
          <p
            id={folderSupportDescriptionId}
            className="text-xs text-muted-foreground"
          >
            {nativePathLeasesSupported
              ? "Existing folders are opened through the desktop picker. Unsloth never sends their paths from the interface."
              : isTauri
                ? "Existing folders need the managed desktop backend. Managed workspaces are still available."
                : "Existing folders are available in the desktop app. Managed workspaces are still available here."}
          </p>
          {workspaceMode === "folder" && folder ? (
            <div className="rounded-xl bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
              <span className="font-medium text-foreground">
                {folder.displayName}
              </span>{" "}
              is the live working directory for every chat and code tool in this
              project. It is not automatically indexed as a Source, and deleting
              the Unsloth project will never delete this folder.
            </div>
          ) : null}
        </section>
        <ProjectSourceDropzone
          staged={staged}
          onChange={controller.setSources}
          disabled={busy}
          onPendingChange={controller.setStagingSources}
        />
        <DialogFooter className="flex-wrap gap-2 sm:justify-end">
          <Button
            type="button"
            variant="ghost"
            disabled={busy}
            onClick={controller.clickCancel}
          >
            Cancel
          </Button>
          <Button
            type="button"
            onClick={() => void controller.submit()}
            disabled={controller.createDisabled()}
          >
            {busy
              ? "Creating…"
              : pickingFolder
                ? "Choosing folder…"
                : stagingDrop
                  ? "Adding sources…"
                  : submitLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
