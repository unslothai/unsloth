// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useNavigate } from "@tanstack/react-router";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { toast } from "@/lib/toast";

import { createChatProject } from "../hooks/use-chat-projects";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";

// Create-project dialog usable from the composer + menu. Creating opens the new
// project straight away rather than dropping the user on the projects list.
export function NewProjectDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const navigate = useNavigate();
  const [name, setName] = useState("");

  async function commitCreate() {
    const trimmed = name.trim();
    if (!trimmed) return;
    try {
      const project = await createChatProject(trimmed);
      onOpenChange(false);
      setName("");
      const runtime = useChatRuntimeStore.getState();
      runtime.setActiveThreadId(null);
      runtime.setActiveProjectId(project.id);
      navigate({ to: "/chat", search: { project: project.id } });
    } catch (err) {
      toast.error("Failed to create project", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!next) setName("");
        onOpenChange(next);
      }}
    >
      <DialogContent className="corner-squircle dialog-soft-surface sm:max-w-md">
        <DialogHeader>
          <DialogTitle>New project</DialogTitle>
        </DialogHeader>
        <Input
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              void commitCreate();
            }
          }}
          autoFocus={true}
          maxLength={120}
          placeholder="Project name"
          aria-label="Project name"
          className="focus-visible:border-input focus-visible:ring-0"
        />
        <DialogFooter className="flex-wrap gap-2 sm:justify-end">
          <Button
            type="button"
            variant="ghost"
            onClick={() => onOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            type="button"
            onClick={() => void commitCreate()}
            disabled={!name.trim()}
          >
            Create
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
