// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { PlusIcon, Trash2Icon, PencilIcon, CheckIcon, XIcon } from "lucide-react";
import { type FC, useCallback, useState } from "react";
import { db, useLiveQuery } from "../db";
import type { MemoryRecord } from "../types";

const MAX_MEMORIES = 20;

export const MemoryPanel: FC = () => {
  const memories = useLiveQuery(
    () => db.memory.orderBy("createdAt").toArray(),
    [],
  );
  const [adding, setAdding] = useState(false);
  const [newContent, setNewContent] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState("");

  const estimatedTokens = (memories ?? [])
    .filter((m) => m.enabled)
    .reduce((sum, m) => sum + Math.ceil(m.content.length / 4), 0);

  const handleAdd = useCallback(async () => {
    if (!newContent.trim()) return;
    await db.memory.add({
      id: crypto.randomUUID(),
      content: newContent.trim(),
      enabled: true,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    });
    setNewContent("");
    setAdding(false);
  }, [newContent]);

  const handleToggle = useCallback(async (id: string, enabled: boolean) => {
    await db.memory.update(id, { enabled, updatedAt: Date.now() });
  }, []);

  const handleDelete = useCallback(async (id: string) => {
    await db.memory.delete(id);
  }, []);

  const handleStartEdit = useCallback((m: MemoryRecord) => {
    setEditingId(m.id);
    setEditContent(m.content);
  }, []);

  const handleSaveEdit = useCallback(async () => {
    if (!editingId || !editContent.trim()) return;
    await db.memory.update(editingId, {
      content: editContent.trim(),
      updatedAt: Date.now(),
    });
    setEditingId(null);
    setEditContent("");
  }, [editingId, editContent]);

  const items = memories ?? [];

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-xs text-muted-foreground">
          Persistent context injected into every conversation.
        </p>
        {estimatedTokens > 0 && (
          <span
            className={`text-xs tabular-nums ${estimatedTokens > 512 ? "text-amber-500" : "text-muted-foreground"}`}
          >
            ~{estimatedTokens} tokens
          </span>
        )}
      </div>
      <div className="space-y-2">
        {items.map((m) => (
          <div
            key={m.id}
            className="flex items-start gap-2 rounded-md border p-2"
          >
            <Switch
              checked={m.enabled}
              onCheckedChange={(v) => handleToggle(m.id, v)}
              className="mt-0.5 scale-75"
            />
            <div className="min-w-0 flex-1">
              {editingId === m.id ? (
                <div className="space-y-1">
                  <Textarea
                    value={editContent}
                    onChange={(e) => setEditContent(e.target.value)}
                    className="min-h-[3rem] text-xs"
                    rows={2}
                  />
                  <div className="flex gap-1">
                    <Button size="sm" variant="ghost" className="h-6 px-2" onClick={handleSaveEdit}>
                      <CheckIcon className="size-3" />
                    </Button>
                    <Button size="sm" variant="ghost" className="h-6 px-2" onClick={() => setEditingId(null)}>
                      <XIcon className="size-3" />
                    </Button>
                  </div>
                </div>
              ) : (
                <p className="text-xs leading-relaxed">{m.content}</p>
              )}
            </div>
            {editingId !== m.id && (
              <div className="flex shrink-0 gap-0.5">
                <button
                  type="button"
                  onClick={() => handleStartEdit(m)}
                  className="p-1 text-muted-foreground hover:text-foreground"
                >
                  <PencilIcon className="size-3" />
                </button>
                <button
                  type="button"
                  onClick={() => handleDelete(m.id)}
                  className="p-1 text-muted-foreground hover:text-destructive"
                >
                  <Trash2Icon className="size-3" />
                </button>
              </div>
            )}
          </div>
        ))}
      </div>

      {adding ? (
        <div className="space-y-2">
          <Textarea
            value={newContent}
            onChange={(e) => setNewContent(e.target.value)}
            placeholder="e.g., I train on medical data, My GPU is A100 40GB..."
            className="min-h-[3rem] text-xs"
            rows={2}
            autoFocus
          />
          <div className="flex gap-2">
            <Button size="sm" onClick={handleAdd} disabled={!newContent.trim()}>
              Save
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                setAdding(false);
                setNewContent("");
              }}
            >
              Cancel
            </Button>
          </div>
        </div>
      ) : (
        <Button
          size="sm"
          variant="outline"
          className="w-full"
          onClick={() => setAdding(true)}
          disabled={items.length >= MAX_MEMORIES}
        >
          <PlusIcon className="mr-1 size-3" />
          Add memory ({items.length}/{MAX_MEMORIES})
        </Button>
      )}
    </div>
  );
};
