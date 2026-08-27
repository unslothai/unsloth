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
  if (typeof window === "undefined") return "";
  return window.location.pathname + window.location.search;
}

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
  const supportDescriptionId = useId();
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
  const state = useSyncExternalStore(
    controller.subscribe,
    controller.getState,
    controller.getState,
  );

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
            value={state.name}
            onChange={(event) => controller.setName(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                void controller.submit();
              }
            }}
            autoFocus={true}
            disabled={state.busy}
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
              Create an Unsloth-managed workspace or work directly in an
              existing local folder.
            </p>
          </div>
          <fieldset
            className="grid grid-cols-1 gap-2 sm:grid-cols-2"
            aria-describedby={supportDescriptionId}
          >
            <legend className="sr-only">Project workspace</legend>
            <Button
              type="button"
              variant={
                state.workspaceMode === "managed" ? "default" : "outline"
              }
              disabled={state.busy}
              aria-pressed={state.workspaceMode === "managed"}
              onClick={controller.clickManagedWorkspace}
            >
              Managed workspace
            </Button>
            <Button
              type="button"
              variant={state.workspaceMode === "folder" ? "default" : "outline"}
              disabled={controller.folderPickerDisabled()}
              aria-pressed={state.workspaceMode === "folder"}
              onClick={() => void controller.clickExistingFolder()}
              title={
                nativePathLeasesSupported
                  ? "Choose an existing local folder"
                  : isTauri
                    ? "Existing folders require the managed desktop backend"
                    : "Existing folders require the desktop app"
              }
            >
              {state.pickingFolder
                ? "Choosing..."
                : state.folder?.displayName || "Use existing folder"}
            </Button>
          </fieldset>
          <p
            id={supportDescriptionId}
            className="text-xs text-muted-foreground"
          >
            {nativePathLeasesSupported
              ? "Folder access is granted through the desktop picker."
              : "Existing folders are available in the desktop app; managed workspaces remain available here."}
          </p>
          {state.workspaceMode === "folder" && state.folder ? (
            <div className="space-y-1 rounded-xl bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
              <code className="block break-all text-foreground">
                {state.folder.displayPath}
              </code>
              <p>
                This is the live working directory for project chats and code
                tools. It is not indexed as a Source, and deleting the Unsloth
                project never deletes this folder.
              </p>
            </div>
          ) : null}
        </section>

        <ProjectSourceDropzone
          staged={state.sources}
          onChange={controller.setSources}
          disabled={state.busy}
          onPendingChange={controller.setStagingSources}
        />
        <DialogFooter className="flex-wrap gap-2 sm:justify-end">
          <Button
            type="button"
            variant="ghost"
            disabled={state.busy}
            onClick={controller.clickCancel}
          >
            Cancel
          </Button>
          <Button
            type="button"
            onClick={() => void controller.submit()}
            disabled={controller.createDisabled()}
          >
            {state.busy
              ? "Creating..."
              : state.stagingSources
                ? "Adding sources..."
                : submitLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
