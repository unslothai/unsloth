// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import { CheckIcon, PencilIcon, PlusIcon, PlayIcon, Trash2Icon, XIcon, GripVerticalIcon } from "lucide-react";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { toast } from "sonner";
import { db } from "../db";
import type { PromptEntry, PromptListEntry } from "../db";
import { useLiveQuery } from "../db";
import { useBenchmarkStore } from "../stores/use-benchmark-store";

// ── helpers ─────────────────────────────────────────────────────────────────

function newId(): string {
  return crypto.randomUUID().replace(/-/g, "").slice(0, 12);
}

function now(): number {
  return Date.now();
}

// ── Prompt card ──────────────────────────────────────────────────────────────

function PromptCard({
  entry,
  onUse,
}: {
  entry: PromptEntry;
  onUse: (text: string) => void;
}): ReactElement {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(entry.name);
  const [text, setText] = useState(entry.text);

  const handleSave = useCallback(async () => {
    const trimName = name.trim();
    const trimText = text.trim();
    if (!trimText) return;
    await db.promptEntries.update(entry.id, {
      name: trimName || "Untitled Prompt",
      text: trimText,
      updatedAt: now(),
    });
    setEditing(false);
  }, [entry.id, name, text]);

  const handleDelete = useCallback(async () => {
    await db.promptEntries.delete(entry.id);
  }, [entry.id]);

  if (editing) {
    return (
      <div className="rounded-xl border border-primary/30 bg-primary/5 p-3 flex flex-col gap-2">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Prompt name..."
          className="w-full rounded-lg border border-border bg-background px-3 py-1.5 text-sm outline-none focus:ring-1 focus:ring-primary"
        />
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={4}
          placeholder="Prompt text..."
          className="w-full resize-y rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-primary"
        />
        <div className="flex gap-2 justify-end">
          <Button size="sm" variant="ghost" onClick={() => { setName(entry.name); setText(entry.text); setEditing(false); }}>
            <XIcon className="size-3.5 mr-1" />Cancel
          </Button>
          <Button size="sm" onClick={handleSave}>
            <CheckIcon className="size-3.5 mr-1" />Save
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="group rounded-xl border border-border bg-card p-3 flex flex-col gap-1.5 hover:border-border/80 transition-colors">
      <div className="flex items-center gap-2">
        <span className="font-medium text-sm flex-1 truncate">{entry.name}</span>
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            type="button"
            onClick={() => onUse(entry.text)}
            className="flex items-center gap-1 rounded-lg bg-primary px-2.5 py-1 text-xs font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
            title="Load into composer"
          >
            <PlayIcon className="size-3" />Use
          </button>
          <button
            type="button"
            onClick={() => setEditing(true)}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
            title="Edit"
          >
            <PencilIcon className="size-3.5" />
          </button>
          <button
            type="button"
            onClick={handleDelete}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors"
            title="Delete"
          >
            <Trash2Icon className="size-3.5" />
          </button>
        </div>
      </div>
      <p className="text-xs text-muted-foreground line-clamp-2 leading-relaxed">{entry.text}</p>
    </div>
  );
}

// ── New prompt form ──────────────────────────────────────────────────────────

function NewPromptForm({ onClose }: { onClose: () => void }): ReactElement {
  const [name, setName] = useState("");
  const [text, setText] = useState("");

  const handleSave = useCallback(async () => {
    const trimText = text.trim();
    if (!trimText) return;
    await db.promptEntries.add({
      id: newId(),
      name: name.trim() || "Untitled Prompt",
      text: trimText,
      createdAt: now(),
      updatedAt: now(),
    });
    onClose();
  }, [name, text, onClose]);

  return (
    <div className="rounded-xl border border-primary/40 bg-primary/5 p-3 flex flex-col gap-2">
      <p className="text-xs font-semibold text-primary">New Prompt</p>
      <input
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Prompt name (optional)..."
        autoFocus
        className="w-full rounded-lg border border-border bg-background px-3 py-1.5 text-sm outline-none focus:ring-1 focus:ring-primary"
      />
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={4}
        placeholder="Write your prompt here..."
        className="w-full resize-y rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-primary"
      />
      <div className="flex gap-2 justify-end">
        <Button size="sm" variant="ghost" onClick={onClose}>
          <XIcon className="size-3.5 mr-1" />Cancel
        </Button>
        <Button size="sm" onClick={handleSave} disabled={!text.trim()}>
          <CheckIcon className="size-3.5 mr-1" />Save Prompt
        </Button>
      </div>
    </div>
  );
}

// ── Prompt list card ─────────────────────────────────────────────────────────

function PromptListCard({
  entry,
  onRun,
}: {
  entry: PromptListEntry;
  onRun: (items: string[]) => void;
}): ReactElement {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(entry.name);
  const [items, setItems] = useState<string[]>(entry.items);

  const handleSave = useCallback(async () => {
    const filtered = items.filter((t) => t.trim());
    if (filtered.length === 0) return;
    await db.promptLists.update(entry.id, {
      name: name.trim() || "Untitled List",
      items: filtered,
      updatedAt: now(),
    });
    setEditing(false);
  }, [entry.id, name, items]);

  const handleDelete = useCallback(async () => {
    await db.promptLists.delete(entry.id);
  }, [entry.id]);

  const addItem = useCallback(() => setItems((prev) => [...prev, ""]), []);
  const removeItem = useCallback((i: number) => setItems((prev) => prev.filter((_, idx) => idx !== i)), []);
  const updateItem = useCallback((i: number, val: string) =>
    setItems((prev) => prev.map((v, idx) => (idx === i ? val : v))), []);

  if (editing) {
    return (
      <div className="rounded-xl border border-primary/30 bg-primary/5 p-3 flex flex-col gap-2">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="List name..."
          className="w-full rounded-lg border border-border bg-background px-3 py-1.5 text-sm outline-none focus:ring-1 focus:ring-primary"
        />
        <p className="text-xs font-semibold text-muted-foreground mt-1">Prompts (sent in order)</p>
        <div className="flex flex-col gap-1.5">
          {items.map((item, i) => (
            <div key={i} className="flex items-start gap-1.5">
              <GripVerticalIcon className="size-4 mt-2 shrink-0 text-muted-foreground/40 cursor-grab" />
              <span className="text-xs font-medium text-muted-foreground mt-2 w-5 shrink-0">{i + 1}.</span>
              <textarea
                value={item}
                onChange={(e) => updateItem(i, e.target.value)}
                rows={2}
                placeholder={`Prompt ${i + 1}...`}
                className="flex-1 resize-y rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-primary"
              />
              <button
                type="button"
                onClick={() => removeItem(i)}
                className="flex h-7 w-7 mt-1 items-center justify-center rounded-lg text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors shrink-0"
                title="Remove"
              >
                <XIcon className="size-3.5" />
              </button>
            </div>
          ))}
        </div>
        <button
          type="button"
          onClick={addItem}
          className="flex items-center gap-1.5 text-xs text-primary hover:text-primary/80 transition-colors mt-1"
        >
          <PlusIcon className="size-3.5" />Add prompt
        </button>
        <div className="flex gap-2 justify-end">
          <Button size="sm" variant="ghost" onClick={() => {
            setName(entry.name);
            setItems(entry.items);
            setEditing(false);
          }}>
            <XIcon className="size-3.5 mr-1" />Cancel
          </Button>
          <Button size="sm" onClick={handleSave} disabled={items.filter((t) => t.trim()).length === 0}>
            <CheckIcon className="size-3.5 mr-1" />Save List
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="group rounded-xl border border-border bg-card p-3 flex flex-col gap-1.5 hover:border-border/80 transition-colors">
      <div className="flex items-center gap-2">
        <span className="font-medium text-sm flex-1 truncate">{entry.name}</span>
        <span className="text-xs text-muted-foreground shrink-0">{entry.items.length} prompt{entry.items.length !== 1 ? "s" : ""}</span>
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            type="button"
            onClick={() => onRun(entry.items)}
            className="flex items-center gap-1 rounded-lg bg-primary px-2.5 py-1 text-xs font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
            title="Run all prompts as benchmark"
          >
            <PlayIcon className="size-3" />Run
          </button>
          <button
            type="button"
            onClick={() => setEditing(true)}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
            title="Edit"
          >
            <PencilIcon className="size-3.5" />
          </button>
          <button
            type="button"
            onClick={handleDelete}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors"
            title="Delete"
          >
            <Trash2Icon className="size-3.5" />
          </button>
        </div>
      </div>
      <div className="flex flex-col gap-0.5 mt-0.5">
        {entry.items.slice(0, 3).map((item, i) => (
          <p key={i} className="text-xs text-muted-foreground flex gap-1.5 leading-relaxed">
            <span className="text-muted-foreground/50 shrink-0">{i + 1}.</span>
            <span className="line-clamp-1">{item}</span>
          </p>
        ))}
        {entry.items.length > 3 && (
          <p className="text-xs text-muted-foreground/50 ml-4">
            +{entry.items.length - 3} more
          </p>
        )}
      </div>
    </div>
  );
}

// ── New prompt list form ─────────────────────────────────────────────────────

function NewPromptListForm({ onClose }: { onClose: () => void }): ReactElement {
  const [name, setName] = useState("");
  const [items, setItems] = useState<string[]>(["", ""]);

  const handleSave = useCallback(async () => {
    const filtered = items.filter((t) => t.trim());
    if (filtered.length === 0) return;
    await db.promptLists.add({
      id: newId(),
      name: name.trim() || "Untitled List",
      items: filtered,
      createdAt: now(),
      updatedAt: now(),
    });
    onClose();
  }, [name, items, onClose]);

  const addItem = useCallback(() => setItems((prev) => [...prev, ""]), []);
  const removeItem = useCallback((i: number) => setItems((prev) => prev.filter((_, idx) => idx !== i)), []);
  const updateItem = useCallback((i: number, val: string) =>
    setItems((prev) => prev.map((v, idx) => (idx === i ? val : v))), []);

  return (
    <div className="rounded-xl border border-primary/40 bg-primary/5 p-3 flex flex-col gap-2">
      <p className="text-xs font-semibold text-primary">New Prompt List</p>
      <input
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="List name (e.g. Snake game conversation)..."
        autoFocus
        className="w-full rounded-lg border border-border bg-background px-3 py-1.5 text-sm outline-none focus:ring-1 focus:ring-primary"
      />
      <p className="text-xs font-semibold text-muted-foreground">
        Prompts — sent to each model in order, one at a time
      </p>
      <div className="flex flex-col gap-1.5">
        {items.map((item, i) => (
          <div key={i} className="flex items-start gap-1.5">
            <span className="text-xs font-medium text-muted-foreground mt-2 w-5 shrink-0">{i + 1}.</span>
            <textarea
              value={item}
              onChange={(e) => updateItem(i, e.target.value)}
              rows={2}
              placeholder={`Prompt ${i + 1}...`}
              className="flex-1 resize-y rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-primary"
            />
            <button
              type="button"
              onClick={() => removeItem(i)}
              disabled={items.length <= 1}
              className="flex h-7 w-7 mt-1 items-center justify-center rounded-lg text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors shrink-0 disabled:opacity-30 disabled:cursor-not-allowed"
              title="Remove"
            >
              <XIcon className="size-3.5" />
            </button>
          </div>
        ))}
      </div>
      <button
        type="button"
        onClick={addItem}
        className="flex items-center gap-1.5 text-xs text-primary hover:text-primary/80 transition-colors"
      >
        <PlusIcon className="size-3.5" />Add another prompt
      </button>
      <div className="flex gap-2 justify-end mt-1">
        <Button size="sm" variant="ghost" onClick={onClose}>
          <XIcon className="size-3.5 mr-1" />Cancel
        </Button>
        <Button
          size="sm"
          onClick={handleSave}
          disabled={items.filter((t) => t.trim()).length === 0}
        >
          <CheckIcon className="size-3.5 mr-1" />Save Prompt List
        </Button>
      </div>
    </div>
  );
}

// ── Main dialog ──────────────────────────────────────────────────────────────

type Tab = "prompts" | "lists";

export function PromptStorageDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}): ReactElement {
  const [activeTab, setActiveTab] = useState<Tab>("prompts");
  const [showNewPrompt, setShowNewPrompt] = useState(false);
  const [showNewList, setShowNewList] = useState(false);

  const promptEntries = useLiveQuery(() =>
    db.promptEntries.orderBy("createdAt").reverse().toArray(), []);
  const promptLists = useLiveQuery(() =>
    db.promptLists.orderBy("createdAt").reverse().toArray(), []);

  const { benchmarkMode, benchmarkSelectedModelIds, benchmarkSendFn, setPendingComposerText } =
    useBenchmarkStore((s) => ({
      benchmarkMode: s.benchmarkMode,
      benchmarkSelectedModelIds: s.benchmarkSelectedModelIds,
      benchmarkSendFn: s.benchmarkSendFn,
      setPendingComposerText: s.setPendingComposerText,
    }));

  const handleUsePrompt = useCallback(
    (text: string) => {
      setPendingComposerText(text);
      onOpenChange(false);
    },
    [setPendingComposerText, onOpenChange],
  );

  const handleRunList = useCallback(
    (items: string[]) => {
      if (!benchmarkMode) {
        toast.warning("Enable benchmark mode first", {
          description: "Turn on Benchmark mode and select models before running a prompt list.",
        });
        return;
      }
      if (benchmarkSelectedModelIds.length === 0) {
        toast.warning("No models selected", {
          description: "Select at least one model in the benchmark model picker.",
        });
        return;
      }
      if (!benchmarkSendFn) {
        toast.error("Benchmark runner not ready");
        return;
      }
      benchmarkSendFn(items);
      onOpenChange(false);
    },
    [benchmarkMode, benchmarkSelectedModelIds, benchmarkSendFn, onOpenChange],
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl max-h-[80vh] flex flex-col gap-0 p-0 overflow-hidden">
        <DialogHeader className="px-5 pt-5 pb-0 shrink-0">
          <DialogTitle className="text-base font-semibold">Prompt Storage</DialogTitle>
        </DialogHeader>

        {/* Tabs */}
        <div className="flex gap-1 px-5 pt-3 pb-0 shrink-0">
          {(["prompts", "lists"] as Tab[]).map((tab) => (
            <button
              key={tab}
              type="button"
              onClick={() => setActiveTab(tab)}
              className={cn(
                "rounded-lg px-3 py-1.5 text-xs font-medium transition-colors",
                activeTab === tab
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground",
              )}
            >
              {tab === "prompts" ? "Saved Prompts" : "Prompt Lists"}
            </button>
          ))}
        </div>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto px-5 py-3 flex flex-col gap-2">
          {activeTab === "prompts" && (
            <>
              {/* New prompt trigger */}
              {!showNewPrompt ? (
                <button
                  type="button"
                  onClick={() => setShowNewPrompt(true)}
                  className="flex items-center gap-2 rounded-xl border border-dashed border-border/60 px-3 py-2 text-sm text-muted-foreground hover:border-primary/40 hover:text-primary transition-colors"
                >
                  <PlusIcon className="size-4" />New Prompt
                </button>
              ) : (
                <NewPromptForm onClose={() => setShowNewPrompt(false)} />
              )}

              {/* Prompt list */}
              {promptEntries && promptEntries.length > 0 ? (
                promptEntries.map((entry) => (
                  <PromptCard key={entry.id} entry={entry} onUse={handleUsePrompt} />
                ))
              ) : (
                !showNewPrompt && (
                  <div className="flex flex-col items-center justify-center py-10 text-center text-muted-foreground gap-2">
                    <p className="text-sm">No saved prompts yet</p>
                    <p className="text-xs">Click "New Prompt" to save a prompt for quick reuse</p>
                  </div>
                )
              )}
            </>
          )}

          {activeTab === "lists" && (
            <>
              {/* New list trigger */}
              {!showNewList ? (
                <button
                  type="button"
                  onClick={() => setShowNewList(true)}
                  className="flex items-center gap-2 rounded-xl border border-dashed border-border/60 px-3 py-2 text-sm text-muted-foreground hover:border-primary/40 hover:text-primary transition-colors"
                >
                  <PlusIcon className="size-4" />New Prompt List
                </button>
              ) : (
                <NewPromptListForm onClose={() => setShowNewList(false)} />
              )}

              {/* Prompt lists */}
              {promptLists && promptLists.length > 0 ? (
                promptLists.map((entry) => (
                  <PromptListCard key={entry.id} entry={entry} onRun={handleRunList} />
                ))
              ) : (
                !showNewList && (
                  <div className="flex flex-col items-center justify-center py-10 text-center text-muted-foreground gap-2">
                    <p className="text-sm">No prompt lists yet</p>
                    <p className="text-xs opacity-80">A prompt list sends multiple prompts in sequence — great for multi-turn conversations across models</p>
                  </div>
                )
              )}
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
