// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Trash2Icon, PencilIcon, CopyIcon } from "lucide-react";
import { type FC, useCallback, useState } from "react";
import { db, useLiveQuery } from "../db";
import type { PromptRecord } from "../types";

export const PromptLibrarySheet: FC<{
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onInsertPrompt?: (content: string) => void;
}> = ({ open, onOpenChange, onInsertPrompt }) => {
  const prompts = useLiveQuery(
    () => db.prompts.orderBy("createdAt").reverse().toArray(),
    [],
  );
  const [editing, setEditing] = useState<PromptRecord | null>(null);
  const [name, setName] = useState("");
  const [content, setContent] = useState("");
  const [tags, setTags] = useState("");

  const handleStartEdit = useCallback((p: PromptRecord) => {
    setEditing(p);
    setName(p.name);
    setContent(p.content);
    setTags(p.tags.join(", "));
  }, []);

  const handleSave = useCallback(async () => {
    if (!name.trim() || !content.trim()) return;
    const variables: string[] = [];
    const tagList = tags
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean);

    if (editing) {
      await db.prompts.update(editing.id, {
        name: name.trim(),
        content: content.trim(),
        variables,
        tags: tagList,
      });
    } else {
      await db.prompts.add({
        id: crypto.randomUUID(),
        name: name.trim(),
        content: content.trim(),
        variables,
        tags: tagList,
        createdAt: Date.now(),
      });
    }
    setEditing(null);
    setName("");
    setContent("");
    setTags("");
  }, [name, content, tags, editing]);

  const handleDelete = useCallback(async (id: string) => {
    await db.prompts.delete(id);
  }, []);

  const handleInsert = useCallback(
    (p: PromptRecord) => {
      onInsertPrompt?.(p.content);
      onOpenChange(false);
    },
    [onInsertPrompt, onOpenChange],
  );

  const items = prompts ?? [];
  const isEditing = editing !== null || name !== "" || content !== "";

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-[24rem] overflow-y-auto">
        <SheetHeader>
          <SheetTitle>Prompt Library</SheetTitle>
          <SheetDescription>
            Reusable prompt templates. Copied to clipboard on select.
          </SheetDescription>
        </SheetHeader>
        <div className="mt-4 space-y-4">
          {/* Editor */}
          <div className="space-y-2 rounded-md border p-3">
            <Input
              placeholder="Prompt name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="text-sm"
            />
            <Textarea
              placeholder="e.g., Evaluate the current model on GSM8K..."
              value={content}
              onChange={(e) => setContent(e.target.value)}
              className="min-h-[6rem] text-sm"
              rows={4}
            />
            <Input
              placeholder="Tags (comma separated)"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              className="text-sm"
            />
            <div className="flex gap-2">
              <Button size="sm" onClick={handleSave} disabled={!name.trim() || !content.trim()}>
                {editing ? "Update" : "Save"}
              </Button>
              {isEditing && (
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    setEditing(null);
                    setName("");
                    setContent("");
                    setTags("");
                  }}
                >
                  Cancel
                </Button>
              )}
            </div>
          </div>

          {/* List */}
          <div className="space-y-2">
            {items.map((p) => (
              <div
                key={p.id}
                className="group rounded-md border p-3 transition-colors hover:bg-accent/50"
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">{p.name}</span>
                  <div className="flex gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                    <button
                      type="button"
                      onClick={() => handleInsert(p)}
                      className="p-1 text-muted-foreground hover:text-primary"
                      title="Copy to clipboard"
                    >
                      <CopyIcon className="size-3" />
                    </button>
                    <button
                      type="button"
                      onClick={() => handleStartEdit(p)}
                      className="p-1 text-muted-foreground hover:text-foreground"
                    >
                      <PencilIcon className="size-3" />
                    </button>
                    <button
                      type="button"
                      onClick={() => handleDelete(p.id)}
                      className="p-1 text-muted-foreground hover:text-destructive"
                    >
                      <Trash2Icon className="size-3" />
                    </button>
                  </div>
                </div>
                <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">
                  {p.content}
                </p>
                {p.tags.length > 0 && (
                  <div className="mt-1 flex gap-1">
                    {p.tags.map((tag) => (
                      <span
                        key={tag}
                        className="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {items.length === 0 && !isEditing && (
              <p className="py-4 text-center text-xs text-muted-foreground">
                No saved prompts yet. Create one above.
              </p>
            )}
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
};
