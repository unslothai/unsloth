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
  linkStagedFolders,
  ProjectFolderPicker,
  type StagedFolder,
} from "@/features/rag";
import {
  ProjectSourceDropzone,
  type StagedSource,
  uploadStagedSources,
} from "@/features/rag/components/project-source-dropzone";
import { toast } from "@/lib/toast";
import { Folder02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import { createChatProject } from "../hooks/use-chat-projects";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { ProjectRecord } from "../types";
import {
  imeOwnsInputKeydown,
  inputImeHandlers,
  newInputImeState,
} from "../utils/composer-preferences";

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
  const [name, setName] = useState("");
  const [staged, setStaged] = useState<StagedSource[]>([]);
  const [folders, setFolders] = useState<StagedFolder[]>([]);
  const [busy, setBusy] = useState(false);
  // A desktop drop reaches `staged` only after native registration; wait before creating.
  const [stagingDrop, setStagingDrop] = useState(false);
  const [pickingFolder, setPickingFolder] = useState(false);
  const nameImeRef = useRef(newInputImeState());
  const nameImeHandlers = {
    onFocus: () => inputImeHandlers(nameImeRef.current).onFocus(),
    onBlur: () => inputImeHandlers(nameImeRef.current).onBlur(),
    onCompositionStart: () =>
      inputImeHandlers(nameImeRef.current).onCompositionStart(),
    onCompositionEnd: (event: { timeStamp: number }) =>
      inputImeHandlers(nameImeRef.current).onCompositionEnd(event),
  };
  // Uploads outlive this component, so a slow one must not yank the user to the new project after
  // they have navigated away.
  const mounted = useRef(true);
  useEffect(() => {
    // Set on setup too: StrictMode replays setup/cleanup/setup.
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  function reset() {
    setName("");
    setStaged([]);
    setFolders([]);
    setStagingDrop(false);
    setPickingFolder(false);
  }

  // Every close path routes here: a leftover draft would resurface on the next project.
  function close() {
    if (busy) return;
    reset();
    onOpenChange(false);
  }

  async function commitCreate() {
    const trimmed = name.trim();
    if (!trimmed || busy || stagingDrop || pickingFolder) return;
    setBusy(true);
    const origin = currentRoute();
    try {
      const project = await createChatProject(trimmed);
      // Folders first: their leases expire in minutes, and uploads can be slow.
      await linkStagedFolders(project.id, folders);
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
      toast.error("Failed to create project", {
        description: err instanceof Error ? err.message : undefined,
      });
    } finally {
      setBusy(false);
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
      <DialogContent className="corner-squircle dialog-soft-surface gap-5 sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="text-ui-21">{title}</DialogTitle>
        </DialogHeader>
        <div className="flex items-stretch overflow-hidden rounded-[16px] border border-border bg-background transition-colors focus-within:border-ring has-[input:disabled]:opacity-50 dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))]">
          <span className="flex w-10 shrink-0 items-center justify-center pl-1 text-muted-foreground">
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
            {...nameImeHandlers}
            onKeyDown={(e) => {
              if (
                imeOwnsInputKeydown(e, nameImeRef.current, {
                  modifiedEnterSubmits: true,
                })
              )
                return;
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
        <ProjectSourceDropzone
          staged={staged}
          onChange={setStaged}
          disabled={busy}
          onPendingChange={setStagingDrop}
        />
        <ProjectFolderPicker
          folders={folders}
          onChange={setFolders}
          disabled={busy}
          onPendingChange={setPickingFolder}
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
            {busy
              ? "Creating…"
              : stagingDrop || pickingFolder
                ? "Adding sources…"
                : submitLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
