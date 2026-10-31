// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import type { NativeDocumentFolderSelection } from "@/features/native-intents";
import { WorkingFolderPicker } from "./working-folder-picker";
import { LinkedFoldersManager } from "@/features/rag";
import { isModEnter } from "@/features/settings/lib/keyboard-shortcuts";
import { toast } from "@/lib/toast";
import { HugeiconsIcon } from "@hugeicons/react";
import { ProjectIcon } from "@/lib/project-icon";

import { setChatProjectWorkingFolder } from "../api/chat-api";
import {
  renameChatProject,
  updateChatProjectInstructions,
} from "../hooks/use-chat-projects";
import type { ProjectRecord } from "../types";
import {
  imeOwnsInputKeydown,
  inputImeHandlers,
  newInputImeState,
} from "../utils/composer-preferences";

function ProjectWorkingFolder({
  project,
  disabled,
  onPendingChange,
}: {
  project: ProjectRecord;
  disabled: boolean;
  onPendingChange: (pending: boolean) => void;
}) {
  const [folder, setFolder] = useState(project.workingFolder ?? null);
  const seed = `${project.id}\n${project.workingFolder ?? ""}`;
  const [seededFor, setSeededFor] = useState(seed);
  if (seed !== seededFor) {
    setSeededFor(seed);
    setFolder(project.workingFolder ?? null);
  }

  async function apply(selection: NativeDocumentFolderSelection | null) {
    const saved = await setChatProjectWorkingFolder(
      project.id,
      selection?.token ?? null,
    );
    setFolder(saved.workingFolder ?? null);
  }

  return (
    <WorkingFolderPicker
      value={folder}
      onChange={apply}
      disabled={disabled}
      onPendingChange={onPendingChange}
    />
  );
}

/** Name, instructions, working folder and source folders for one project.
 *  Deleting hands back to the caller, which owns the confirmation. */
export function EditProjectDialog({
  project,
  onOpenChange,
  onDelete,
}: {
  project: ProjectRecord | null;
  onOpenChange: (open: boolean) => void;
  onDelete: (project: ProjectRecord) => void;
}) {
  const [name, setName] = useState(project?.name ?? "");
  const [instructions, setInstructions] = useState(project?.instructions ?? "");
  const [busy, setBusy] = useState(false);
  const [folderBusy, setFolderBusy] = useState(false);
  const nameImeRef = useRef(newInputImeState());
  const nameImeHandlers = {
    onFocus: () => inputImeHandlers(nameImeRef.current).onFocus(),
    onBlur: () => inputImeHandlers(nameImeRef.current).onBlur(),
    onCompositionStart: () =>
      inputImeHandlers(nameImeRef.current).onCompositionStart(),
    onCompositionEnd: (event: { timeStamp: number }) =>
      inputImeHandlers(nameImeRef.current).onCompositionEnd(event),
  };
  // Reseed on render, not in an effect: the fields are drafts of whichever project is open, and
  // a stale one would save the last project's text over this one.
  const [seededFor, setSeededFor] = useState(project?.id ?? null);
  if ((project?.id ?? null) !== seededFor) {
    setSeededFor(project?.id ?? null);
    setName(project?.name ?? "");
    setInstructions(project?.instructions ?? "");
    setBusy(false);
    setFolderBusy(false);
  }

  if (!project) return null;
  // Captured once: save() reads it after an await, where the prop may have changed.
  const target = project;

  const trimmedName = name.trim();
  const trimmedInstructions = instructions.trim();
  const nameChanged = trimmedName.length > 0 && trimmedName !== project.name;
  const instructionsChanged =
    trimmedInstructions !== (project.instructions ?? "").trim();
  const dirty = nameChanged || instructionsChanged;

  function close() {
    if (busy || folderBusy) return;
    onOpenChange(false);
  }

  async function save() {
    if (busy || folderBusy || !trimmedName) return;
    if (!dirty) {
      onOpenChange(false);
      return;
    }
    setBusy(true);
    try {
      // Separate writes: a failed rename must not drop the instructions with it.
      if (nameChanged) await renameChatProject(target.id, trimmedName);
      if (instructionsChanged) {
        await updateChatProjectInstructions(target.id, trimmedInstructions);
      }
      onOpenChange(false);
    } catch (err) {
      toast.error("Failed to save project", {
        description: err instanceof Error ? err.message : undefined,
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <Dialog
      open={true}
      onOpenChange={(next) => {
        if (next) return;
        close();
      }}
    >
      <DialogContent
        className="corner-squircle dialog-soft-surface grid-cols-[minmax(0,1fr)] gap-5 sm:max-w-xl"
        {...nameImeHandlers}
        // Enter saves from the name field, which a multi-line instructions box cannot do; the
        // chord saves from either. The menus and confirmations inside portal out of here, so
        // their own keys never reach this.
        onKeyDown={(e) => {
          if (
            imeOwnsInputKeydown(e, nameImeRef.current, {
              modifiedEnterSubmits: true,
            })
          )
            return;
          if (isModEnter(e)) {
            e.preventDefault();
            void save();
          }
        }}
      >
        <DialogHeader>
          <DialogTitle className="text-ui-21">Edit project</DialogTitle>
          <DialogDescription className="sr-only">
            Rename this project, set instructions, choose its working folder,
            and manage source folders for document search.
          </DialogDescription>
        </DialogHeader>
        {/* The same name field the create dialog uses. */}
        <div className="flex items-stretch overflow-hidden rounded-full border border-border bg-background transition-colors focus-within:border-ring has-[input:disabled]:opacity-50 dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))]">
          <span className="flex shrink-0 items-center pl-5 text-muted-foreground">
            <HugeiconsIcon
              icon={ProjectIcon}
              strokeWidth={1.75}
              className="size-5"
            />
          </span>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (
                imeOwnsInputKeydown(e, nameImeRef.current, {
                  modifiedEnterSubmits: true,
                })
              ) {
                e.stopPropagation();
                return;
              }
              if (e.key === "Enter") {
                e.preventDefault();
                // The chord would reach the dialog's handler too and save twice.
                e.stopPropagation();
                void save();
              }
            }}
            autoFocus={true}
            disabled={busy || folderBusy}
            maxLength={120}
            placeholder="Project name"
            aria-label="Project name"
            className="min-w-0 flex-1 bg-transparent py-4 pr-4 pl-3.5 text-base outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed"
          />
        </div>
        <div className="flex flex-col gap-1.5">
          <label
            htmlFor="edit-project-instructions"
            className="text-ui-15 font-medium text-foreground"
          >
            Instructions
          </label>
          <p className="text-ui-13 text-muted-foreground">
            Sent with every chat in this project, ahead of the conversation.
          </p>
          <Textarea
            id="edit-project-instructions"
            value={instructions}
            onChange={(e) => setInstructions(e.target.value)}
            disabled={busy || folderBusy}
            rows={4}
            placeholder="How should the model answer in this project?"
            className="mt-0.5 resize-none rounded-xl"
          />
        </div>
        <ProjectWorkingFolder
          key={project.id}
          project={project}
          disabled={busy}
          onPendingChange={setFolderBusy}
        />
        {/* The Sources tab's own manager, so a folder linked here is linked there. */}
        <div className="flex flex-col gap-1.5">
          <h3 className="text-ui-15 font-medium text-foreground">
            Source folders
          </h3>
          <p className="text-ui-13 text-muted-foreground">
            Keep supported documents in sync for chats to search. Linking does
            not grant access to edit the original files.
          </p>
          <div className="mt-0.5">
            <LinkedFoldersManager
              scope={{ type: "project", id: project.id }}
              variant="card"
            />
          </div>
        </div>
        <DialogFooter className="flex-wrap gap-2 sm:justify-between">
          <Button
            type="button"
            variant="destructive"
            disabled={busy || folderBusy}
            onClick={() => {
              onOpenChange(false);
              onDelete(project);
            }}
          >
            Delete project
          </Button>
          <span className="flex flex-wrap gap-2">
            <Button
              type="button"
              variant="ghost"
              disabled={busy || folderBusy}
              onClick={close}
            >
              Cancel
            </Button>
            <Button
              type="button"
              onClick={() => void save()}
              disabled={busy || folderBusy || !trimmedName}
            >
              {busy ? "Saving…" : "Save"}
            </Button>
          </span>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
