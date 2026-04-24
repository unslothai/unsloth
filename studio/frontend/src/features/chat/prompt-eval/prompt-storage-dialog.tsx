// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import {
  CheckIcon,
  DownloadIcon,
  GripVerticalIcon,
  PencilIcon,
  PlayIcon,
  PlusIcon,
  SearchIcon,
  Trash2Icon,
  UploadIcon,
  XIcon,
} from "lucide-react";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { toast } from "sonner";
import { db } from "../db";
import type { PromptEntry, PromptListEntry } from "../db";
import { useLiveQuery } from "../db";
import { usePromptEvalStore } from "../stores/use-prompt-eval-store";

// ── helpers ─────────────────────────────────────────────────────────────────

function newId(): string {
  return crypto.randomUUID().replace(/-/g, "").slice(0, 12);
}

function now(): number {
  return Date.now();
}

function sanitizeFilename(name: string): string {
  return name.replace(/[\\/:*?"<>|]/g, "_").slice(0, 80) || "export";
}

function downloadBlob(content: string, filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function csvEscape(val: string): string {
  return `"${val.replace(/"/g, '""')}"`;
}

// ── Export helpers ───────────────────────────────────────────────────────────

function exportPromptJsonl(entry: PromptEntry): void {
  downloadBlob(
    JSON.stringify({ name: entry.name, text: entry.text }),
    `${sanitizeFilename(entry.name)}.jsonl`,
    "application/x-ndjson",
  );
}

function exportPromptCsv(entry: PromptEntry): void {
  downloadBlob(
    `name,text\n${csvEscape(entry.name)},${csvEscape(entry.text)}`,
    `${sanitizeFilename(entry.name)}.csv`,
    "text/csv",
  );
}

function exportAllPromptsJsonl(entries: PromptEntry[]): void {
  const lines = entries.map((e) => JSON.stringify({ name: e.name, text: e.text })).join("\n");
  downloadBlob(lines, "prompts.jsonl", "application/x-ndjson");
}

function exportAllPromptsCsv(entries: PromptEntry[]): void {
  const rows = entries.map((e) => `${csvEscape(e.name)},${csvEscape(e.text)}`).join("\n");
  downloadBlob(`name,text\n${rows}`, "prompts.csv", "text/csv");
}

function exportListJsonl(entry: PromptListEntry): void {
  downloadBlob(
    JSON.stringify({ name: entry.name, items: entry.items }),
    `${sanitizeFilename(entry.name)}.jsonl`,
    "application/x-ndjson",
  );
}

function exportAllListsJsonl(entries: PromptListEntry[]): void {
  const lines = entries.map((e) => JSON.stringify({ name: e.name, items: e.items })).join("\n");
  downloadBlob(lines, "prompt-lists.jsonl", "application/x-ndjson");
}

// ── Import helpers ───────────────────────────────────────────────────────────

function parseSimpleCsvRow(line: string): string[] {
  const cells: string[] = [];
  let i = 0;
  while (i < line.length) {
    if (line[i] === '"') {
      i++;
      let cell = "";
      while (i < line.length) {
        if (line[i] === '"' && line[i + 1] === '"') {
          cell += '"';
          i += 2;
        } else if (line[i] === '"') {
          i++;
          break;
        } else {
          cell += line[i++];
        }
      }
      cells.push(cell);
      if (line[i] === ",") i++;
    } else {
      const end = line.indexOf(",", i);
      if (end === -1) {
        cells.push(line.slice(i));
        break;
      } else {
        cells.push(line.slice(i, end));
        i = end + 1;
      }
    }
  }
  return cells;
}

async function importPromptsFromText(text: string, isCsv: boolean): Promise<number> {
  let count = 0;
  if (isCsv) {
    const lines = text.split("\n").slice(1); // skip header row
    for (const raw of lines) {
      const line = raw.trim();
      if (!line) continue;
      const cells = parseSimpleCsvRow(line);
      if (cells.length >= 2 && cells[1]?.trim()) {
        await db.promptEntries.add({
          id: newId(),
          name: cells[0]?.trim() || "Imported",
          text: cells[1].trim(),
          createdAt: now(),
          updatedAt: now(),
        });
        count++;
      }
    }
  } else {
    for (const raw of text.split("\n")) {
      const line = raw.trim();
      if (!line) continue;
      try {
        const obj = JSON.parse(line) as Record<string, unknown>;
        if (typeof obj.text === "string" && obj.text.trim()) {
          await db.promptEntries.add({
            id: newId(),
            name: typeof obj.name === "string" ? obj.name || "Imported" : "Imported",
            text: obj.text.trim(),
            createdAt: now(),
            updatedAt: now(),
          });
          count++;
        }
      } catch {
        /* skip malformed lines */
      }
    }
  }
  return count;
}

async function importListsFromText(text: string): Promise<number> {
  let count = 0;
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    try {
      const obj = JSON.parse(line) as Record<string, unknown>;
      if (Array.isArray(obj.items) && obj.items.length > 0) {
        const items = (obj.items as unknown[]).filter(
          (x): x is string => typeof x === "string" && x.trim().length > 0,
        );
        if (items.length > 0) {
          await db.promptLists.add({
            id: newId(),
            name: typeof obj.name === "string" ? obj.name || "Imported" : "Imported",
            items,
            createdAt: now(),
            updatedAt: now(),
          });
          count++;
        }
      }
    } catch {
      /* skip malformed lines */
    }
  }
  return count;
}

// ── Small export dropdown ────────────────────────────────────────────────────

function ExportDropdown({
  onJsonl,
  onCsv,
}: {
  onJsonl: () => void;
  onCsv?: () => void;
}): ReactElement {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
        title="Export"
      >
        <DownloadIcon className="size-3.5" />
      </button>
      {open && (
        <div className="absolute right-0 top-full mt-1 z-50 min-w-[110px] rounded-lg border border-border bg-popover shadow-md overflow-hidden">
          <button
            type="button"
            onClick={() => { onJsonl(); setOpen(false); }}
            className="w-full text-left px-3 py-2 text-xs hover:bg-accent transition-colors"
          >
            Export JSONL
          </button>
          {onCsv && (
            <button
              type="button"
              onClick={() => { onCsv(); setOpen(false); }}
              className="w-full text-left px-3 py-2 text-xs hover:bg-accent transition-colors"
            >
              Export CSV
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// ── Bulk export dropdown ─────────────────────────────────────────────────────

function BulkExportDropdown({
  onJsonl,
  onCsv,
  disabled,
}: {
  onJsonl: () => void;
  onCsv?: () => void;
  disabled?: boolean;
}): ReactElement {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 rounded-lg border border-border bg-background px-2.5 py-1.5 text-xs font-medium text-foreground hover:bg-accent transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        title="Export all"
      >
        <DownloadIcon className="size-3.5" />
        Export All
      </button>
      {open && (
        <div className="absolute right-0 top-full mt-1 z-50 min-w-[120px] rounded-lg border border-border bg-popover shadow-md overflow-hidden">
          <button
            type="button"
            onClick={() => { onJsonl(); setOpen(false); }}
            className="w-full text-left px-3 py-2 text-xs hover:bg-accent transition-colors"
          >
            Export JSONL
          </button>
          {onCsv && (
            <button
              type="button"
              onClick={() => { onCsv(); setOpen(false); }}
              className="w-full text-left px-3 py-2 text-xs hover:bg-accent transition-colors"
            >
              Export CSV
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// ── Prompt card ──────────────────────────────────────────────────────────────

function PromptCard({
  entry,
  onUse,
  promptEvalMode,
}: {
  entry: PromptEntry;
  onUse: (text: string) => void;
  promptEvalMode?: boolean;
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
            title={promptEvalMode ? "Run through all selected models" : "Load into composer"}
          >
            <PlayIcon className="size-3" />{promptEvalMode ? "Run" : "Use"}
          </button>
          <ExportDropdown
            onJsonl={() => exportPromptJsonl(entry)}
            onCsv={() => exportPromptCsv(entry)}
          />
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
  const removeItem = useCallback(
    (i: number) => setItems((prev) => prev.filter((_, idx) => idx !== i)),
    [],
  );
  const updateItem = useCallback(
    (i: number, val: string) =>
      setItems((prev) => prev.map((v, idx) => (idx === i ? val : v))),
    [],
  );

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
          <Button
            size="sm"
            variant="ghost"
            onClick={() => { setName(entry.name); setItems(entry.items); setEditing(false); }}
          >
            <XIcon className="size-3.5 mr-1" />Cancel
          </Button>
          <Button
            size="sm"
            onClick={handleSave}
            disabled={items.filter((t) => t.trim()).length === 0}
          >
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
        <span className="text-xs text-muted-foreground shrink-0">
          {entry.items.length} prompt{entry.items.length !== 1 ? "s" : ""}
        </span>
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            type="button"
            onClick={() => onRun(entry.items)}
            className="flex items-center gap-1 rounded-lg bg-primary px-2.5 py-1 text-xs font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
            title="Run all prompts as Prompt Eval"
          >
            <PlayIcon className="size-3" />Run
          </button>
          <ExportDropdown onJsonl={() => exportListJsonl(entry)} />
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
  const removeItem = useCallback(
    (i: number) => setItems((prev) => prev.filter((_, idx) => idx !== i)),
    [],
  );
  const updateItem = useCallback(
    (i: number, val: string) =>
      setItems((prev) => prev.map((v, idx) => (idx === i ? val : v))),
    [],
  );

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
  const [searchQuery, setSearchQuery] = useState("");
  const [showSuggestions, setShowSuggestions] = useState(false);
  const importRef = useRef<HTMLInputElement>(null);

  const promptEntries = useLiveQuery(
    () => db.promptEntries.orderBy("createdAt").reverse().toArray(),
    [],
  );
  const promptLists = useLiveQuery(
    () => db.promptLists.orderBy("createdAt").reverse().toArray(),
    [],
  );

  const promptEvalMode = usePromptEvalStore((s) => s.promptEvalMode);
  const promptEvalSelectedModelIds = usePromptEvalStore((s) => s.promptEvalSelectedModelIds);
  const promptEvalSendFn = usePromptEvalStore((s) => s.promptEvalSendFn);
  const setPendingComposerText = usePromptEvalStore((s) => s.setPendingComposerText);

  // Reset search + forms when tab changes
  useEffect(() => {
    setSearchQuery("");
    setShowSuggestions(false);
    setShowNewPrompt(false);
    setShowNewList(false);
  }, [activeTab]);

  // ── Filtering ──────────────────────────────────────────────────────────────

  const filteredPrompts = useMemo(() => {
    const all = promptEntries ?? [];
    if (!searchQuery.trim()) return all;
    const q = searchQuery.toLowerCase();
    return all.filter(
      (e) => e.name.toLowerCase().includes(q) || e.text.toLowerCase().includes(q),
    );
  }, [promptEntries, searchQuery]);

  const filteredLists = useMemo(() => {
    const all = promptLists ?? [];
    if (!searchQuery.trim()) return all;
    const q = searchQuery.toLowerCase();
    return all.filter((e) => e.name.toLowerCase().includes(q));
  }, [promptLists, searchQuery]);

  // Live suggestions (names only, max 7)
  const suggestions = useMemo(() => {
    if (!searchQuery.trim()) return [];
    const q = searchQuery.toLowerCase();
    const source: { name: string }[] =
      activeTab === "prompts" ? (promptEntries ?? []) : (promptLists ?? []);
    return source
      .filter((e) => e.name.toLowerCase().includes(q))
      .slice(0, 7)
      .map((e) => e.name);
  }, [searchQuery, activeTab, promptEntries, promptLists]);

  // ── Handlers ──────────────────────────────────────────────────────────────

  const handleUsePrompt = useCallback(
    (text: string) => {
      // In Prompt Eval mode, single prompts behave like list prompts: fire the
      // eval runner directly instead of just loading into the composer.
      if (promptEvalMode) {
        if (promptEvalSelectedModelIds.length === 0) {
          toast.warning("No models selected", {
            description: "Select at least one model in the Prompt Eval model picker.",
          });
          return;
        }
        if (!promptEvalSendFn) {
          toast.error("Prompt Eval runner not ready");
          return;
        }
        promptEvalSendFn([text]);
        onOpenChange(false);
        return;
      }
      setPendingComposerText(text);
      onOpenChange(false);
    },
    [promptEvalMode, promptEvalSelectedModelIds, promptEvalSendFn, setPendingComposerText, onOpenChange],
  );

  const handleRunList = useCallback(
    (items: string[]) => {
      if (!promptEvalMode) {
        toast.warning("Enable Prompt Eval mode first", {
          description: "Turn on Prompt Eval mode and select models before running a prompt list.",
        });
        return;
      }
      if (promptEvalSelectedModelIds.length === 0) {
        toast.warning("No models selected", {
          description: "Select at least one model in the Prompt Eval model picker.",
        });
        return;
      }
      if (!promptEvalSendFn) {
        toast.error("Prompt Eval runner not ready");
        return;
      }
      promptEvalSendFn(items);
      onOpenChange(false);
    },
    [promptEvalMode, promptEvalSelectedModelIds, promptEvalSendFn, onOpenChange],
  );

  const handleImportFile = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const text = await file.text();
      const isCsv = file.name.toLowerCase().endsWith(".csv");
      let count = 0;
      try {
        if (activeTab === "prompts") {
          count = await importPromptsFromText(text, isCsv);
        } else {
          count = await importListsFromText(text);
        }
        if (count > 0) {
          toast.success(`Imported ${count} item${count !== 1 ? "s" : ""}`);
        } else {
          toast.warning("No items imported", {
            description: "The file may be empty or in an unsupported format.",
          });
        }
      } catch {
        toast.error("Import failed", { description: "Could not parse the file." });
      }
      e.target.value = "";
    },
    [activeTab],
  );

  const handleExportAll = useCallback(
    (format: "jsonl" | "csv") => {
      if (activeTab === "prompts") {
        const entries = promptEntries ?? [];
        if (entries.length === 0) { toast.info("No prompts to export"); return; }
        if (format === "csv") { exportAllPromptsCsv(entries); } else { exportAllPromptsJsonl(entries); }
      } else {
        const lists = promptLists ?? [];
        if (lists.length === 0) { toast.info("No prompt lists to export"); return; }
        exportAllListsJsonl(lists);
      }
    },
    [activeTab, promptEntries, promptLists],
  );

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-3xl max-h-[88vh] flex flex-col gap-0 p-0 overflow-hidden">
        <DialogHeader className="px-6 pt-5 pb-0 shrink-0">
          <DialogTitle className="text-base font-semibold">Prompt Storage</DialogTitle>
          <DialogDescription className="sr-only">
            Save and manage reusable prompts and prompt lists for Prompt Eval runs.
          </DialogDescription>
        </DialogHeader>

        {/* Tabs + import/export toolbar */}
        <div className="flex items-center gap-2 px-6 pt-3 pb-2 shrink-0 flex-wrap">
          <div className="flex gap-1 flex-1 min-w-0">
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

          {/* Import */}
          <input
            ref={importRef}
            type="file"
            accept=".jsonl,.json,.csv"
            className="hidden"
            onChange={handleImportFile}
          />
          <button
            type="button"
            onClick={() => importRef.current?.click()}
            className="flex items-center gap-1.5 rounded-lg border border-border bg-background px-2.5 py-1.5 text-xs font-medium text-foreground hover:bg-accent transition-colors"
            title={activeTab === "prompts" ? "Import prompts from JSONL or CSV" : "Import prompt lists from JSONL"}
          >
            <UploadIcon className="size-3.5" />
            Import
          </button>

          {/* Export All */}
          <BulkExportDropdown
            disabled={(activeTab === "prompts" ? (promptEntries?.length ?? 0) : (promptLists?.length ?? 0)) === 0}
            onJsonl={() => handleExportAll("jsonl")}
            onCsv={activeTab === "prompts" ? () => handleExportAll("csv") : undefined}
          />
        </div>

        {/* Search bar */}
        <div className="px-6 pb-2 shrink-0">
          <div className="relative">
            <SearchIcon className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 size-3.5 text-muted-foreground" />
            <input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onFocus={() => setShowSuggestions(true)}
              onBlur={() => setTimeout(() => setShowSuggestions(false), 150)}
              placeholder={`Search ${activeTab === "prompts" ? "prompts by name or text" : "prompt lists by name"}\u2026`}
              className="w-full rounded-lg border border-border bg-background pl-8 pr-3 py-1.5 text-sm outline-none focus:ring-1 focus:ring-primary"
            />
            {/* Suggestions dropdown */}
            {showSuggestions && searchQuery.trim() !== "" && suggestions.length > 0 && (
              <div className="absolute top-full left-0 right-0 z-50 mt-1 rounded-lg border border-border bg-popover shadow-lg overflow-hidden">
                {suggestions.map((name) => (
                  <button
                    key={name}
                    type="button"
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={() => { setSearchQuery(name); setShowSuggestions(false); }}
                    className="flex w-full items-center gap-2 px-3 py-2 text-sm hover:bg-accent hover:text-accent-foreground transition-colors text-left"
                  >
                    <SearchIcon className="size-3 shrink-0 text-muted-foreground" />
                    <span className="truncate">{name}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Scrollable content */}
        <div className="flex-1 min-h-0 overflow-y-auto px-6 pb-4 flex flex-col gap-2">
          {activeTab === "prompts" && (
            <>
              {!showNewPrompt ? (
                <button
                  type="button"
                  onClick={() => setShowNewPrompt(true)}
                  className="flex items-center gap-2 rounded-xl border border-dashed border-border/60 px-3 py-2.5 text-sm text-muted-foreground hover:border-primary/40 hover:text-primary transition-colors"
                >
                  <PlusIcon className="size-4" />New Prompt
                </button>
              ) : (
                <NewPromptForm onClose={() => setShowNewPrompt(false)} />
              )}

              {filteredPrompts.length > 0 ? (
                filteredPrompts.map((entry) => (
                  <PromptCard key={entry.id} entry={entry} onUse={handleUsePrompt} promptEvalMode={promptEvalMode} />
                ))
              ) : (
                !showNewPrompt && (
                  <div className="flex flex-col items-center justify-center py-12 text-center text-muted-foreground gap-2">
                    {searchQuery.trim() ? (
                      <>
                        <p className="text-sm">No prompts match &ldquo;{searchQuery}&rdquo;</p>
                        <button type="button" onClick={() => setSearchQuery("")} className="text-xs text-primary hover:underline">
                          Clear search
                        </button>
                      </>
                    ) : (
                      <>
                        <p className="text-sm">No saved prompts yet</p>
                        <p className="text-xs">Click &ldquo;New Prompt&rdquo; to save a prompt for quick reuse</p>
                      </>
                    )}
                  </div>
                )
              )}
            </>
          )}

          {activeTab === "lists" && (
            <>
              {!showNewList ? (
                <button
                  type="button"
                  onClick={() => setShowNewList(true)}
                  className="flex items-center gap-2 rounded-xl border border-dashed border-border/60 px-3 py-2.5 text-sm text-muted-foreground hover:border-primary/40 hover:text-primary transition-colors"
                >
                  <PlusIcon className="size-4" />New Prompt List
                </button>
              ) : (
                <NewPromptListForm onClose={() => setShowNewList(false)} />
              )}

              {filteredLists.length > 0 ? (
                filteredLists.map((entry) => (
                  <PromptListCard key={entry.id} entry={entry} onRun={handleRunList} />
                ))
              ) : (
                !showNewList && (
                  <div className="flex flex-col items-center justify-center py-12 text-center text-muted-foreground gap-2">
                    {searchQuery.trim() ? (
                      <>
                        <p className="text-sm">No prompt lists match &ldquo;{searchQuery}&rdquo;</p>
                        <button type="button" onClick={() => setSearchQuery("")} className="text-xs text-primary hover:underline">
                          Clear search
                        </button>
                      </>
                    ) : (
                      <>
                        <p className="text-sm">No prompt lists yet</p>
                        <p className="text-xs opacity-80">
                          A prompt list sends multiple prompts in sequence &mdash; great for multi-turn conversations across models
                        </p>
                      </>
                    )}
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
