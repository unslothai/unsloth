// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useState } from "react";

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
import { LinkedFoldersManager } from "@/features/rag";
import { toast } from "@/lib/toast";
import { Folder02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import {
  renameChatProject,
  updateChatProjectInstructions,
} from "../hooks/use-chat-projects";
import type { ProjectRecord } from "../types";

/** Name, instructions and linked folders for one project. Opened from the folder row's "Edit".
 *  Deleting hands back to the caller, which owns the confirmation. */
export function EditProjectDialog({
  project,
  onOpenChange,
  onDelete,
}: {
  /** The project being edited, or null while the dialog is closed. */
  project: ProjectRecord | null;
  onOpenChange: (open: boolean) => void;
  onDelete: (project: ProjectRecord) => void;
}) {
  const [name, setName] = useState(project?.name ?? "");
  const [instructions, setInstructions] = useState(project?.instructions ?? "");
  const [busy, setBusy] = useState(false);
  // Reseed on render, not in an effect: the fields are drafts of whichever project is open, and
  // a stale one would save the last project's text over this one.
  const [seededFor, setSeededFor] = useState(project?.id ?? null);
  if ((project?.id ?? null) !== seededFor) {
    setSeededFor(project?.id ?? null);
    setName(project?.name ?? "");
    setInstructions(project?.instructions ?? "");
    setBusy(false);
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
    if (busy) return;
    onOpenChange(false);
  }

  async function save() {
    if (busy || !trimmedName) return;
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
        className="corner-squircle dialog-soft-surface gap-5 sm:max-w-lg"
        // Enter saves from the name field, which a multi-line instructions box cannot do; the
        // chord saves from either. The menus and confirmations inside portal out of here, so
        // their own keys never reach this.
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            void save();
          }
        }}
      >
        <DialogHeader>
          <DialogTitle className="text-ui-21">Edit project</DialogTitle>
          <DialogDescription className="sr-only">
            Rename this project, set the instructions its chats follow, and manage the local
            folders it indexes.
          </DialogDescription>
        </DialogHeader>
        {/* The same name field the create dialog uses. */}
        <div className="flex items-stretch overflow-hidden rounded-[16px] border border-border bg-background transition-colors focus-within:border-ring has-[input:disabled]:opacity-50 dark:border-transparent dark:bg-white/[0.06]">
          <span className="flex w-10 shrink-0 items-center justify-center pl-1 text-muted-foreground">
            <HugeiconsIcon
              icon={Folder02Icon}
              strokeWidth={1.75}
              className="size-5"
            />
          </span>
          {/* The rule that separates the icon from the field. The container drops its border in
              dark mode, where --border goes with it, so the rule carries its own tint there. */}
          <span aria-hidden="true" className="my-3 w-px bg-border dark:bg-white/10" />
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                // The chord would reach the dialog's handler too and save twice.
                e.stopPropagation();
                void save();
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
            disabled={busy}
            rows={4}
            placeholder="How should the model answer in this project?"
            className="mt-0.5 resize-none rounded-[16px]"
          />
        </div>
        {/* The Sources tab's own manager, so a folder linked here is linked there. */}
        <div className="flex flex-col gap-1.5">
          <h3 className="text-ui-15 font-medium text-foreground">
            Source folders
          </h3>
          <p className="text-ui-13 text-muted-foreground">
            Indexed and kept in sync, for every chat in this project to search.
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
            disabled={busy}
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
              disabled={busy}
              onClick={close}
            >
              Cancel
            </Button>
            <Button
              type="button"
              onClick={() => void save()}
              disabled={busy || !trimmedName}
            >
              {busy ? "Saving…" : "Save"}
            </Button>
          </span>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
