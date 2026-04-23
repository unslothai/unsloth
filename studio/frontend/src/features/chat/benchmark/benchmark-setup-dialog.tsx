// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { CheckIcon } from "lucide-react";
import { useMemo, useState } from "react";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { BenchmarkConfig, BenchmarkModelEntry, PromptItem } from "./types";

type ModelEntry = {
  id: string;
  name: string;
  tag: string;
  isLora: boolean;
  ggufVariant?: string;
};

export function BenchmarkSetupDialog({
  open,
  onOpenChange,
  onStart,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  onStart: (config: BenchmarkConfig) => void;
}) {
  const models = useChatRuntimeStore((s) => s.models);
  const loras = useChatRuntimeStore((s) => s.loras);
  const maxSeqLength = useChatRuntimeStore((s) => s.params.maxSeqLength);

  const [name, setName] = useState(
    () => `Benchmark ${new Date().toLocaleDateString()}`,
  );
  const [promptText, setPromptText] = useState("");
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  // Pre-map to unified display type — avoids isLora field mismatch between
  // ChatModelSummary and ChatLoraSummary
  const allEntries = useMemo<ModelEntry[]>(
    () => [
      ...models.map((m) => ({
        id: m.id,
        name: m.name,
        tag: m.isGguf ? "GGUF" : m.isLora ? "LoRA" : "Base",
        isLora: m.isLora,
      })),
      ...loras.map((l) => ({
        id: l.id,
        name: l.name,
        tag: "LoRA",
        isLora: true,
      })),
    ],
    [models, loras],
  );

  const prompts = useMemo<PromptItem[]>(
    () =>
      promptText
        .split("\n")
        .map((l) => l.trim())
        .filter(Boolean)
        .map((text) => ({ id: crypto.randomUUID(), text })),
    [promptText],
  );

  function handleImportFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const raw = reader.result as string;
      const lines = raw
        .split("\n")
        .filter(Boolean)
        .map((l) => {
          try {
            const obj = JSON.parse(l) as {
              prompt?: string;
              text?: string;
              category?: string;
            };
            return obj.prompt ?? obj.text ?? l;
          } catch {
            // CSV: take first column
            return l.split(",")[0]?.trim() ?? l;
          }
        });
      setPromptText(lines.join("\n"));
    };
    reader.readAsText(file);
    // Reset so re-importing the same file works
    e.target.value = "";
  }

  function toggle(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function handleStart() {
    if (prompts.length === 0 || selectedIds.size === 0) return;
    const selected = allEntries.filter((e) => selectedIds.has(e.id));
    const benchmarkModels: BenchmarkModelEntry[] = selected.map((e) => ({
      id: e.id,
      isLora: e.isLora,
      ggufVariant: e.ggufVariant,
      displayName: e.name,
    }));
    onStart({
      id: crypto.randomUUID(),
      name: name.trim() || `Benchmark ${new Date().toLocaleDateString()}`,
      prompts,
      models: benchmarkModels,
      maxSeqLength,
    });
    onOpenChange(false);
  }

  const canStart = prompts.length > 0 && selectedIds.size > 0;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>New Benchmark</DialogTitle>
          <DialogDescription>
            Run a prompt list across multiple models. Results are saved as
            grouped threads you can export.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Run name */}
          <div className="space-y-1.5">
            <label className="text-sm font-medium">Run name</label>
            <Input value={name} onChange={(e) => setName(e.target.value)} />
          </div>

          {/* Prompts */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">
                Prompts
                {prompts.length > 0 && (
                  <span className="ml-2 rounded-full bg-primary/10 px-2 py-0.5 text-xs text-primary">
                    {prompts.length}
                  </span>
                )}
              </label>
              <label className="cursor-pointer text-xs text-muted-foreground transition-colors hover:text-foreground">
                Import JSONL / CSV
                <input
                  type="file"
                  accept=".jsonl,.csv,.txt"
                  className="sr-only"
                  onChange={handleImportFile}
                />
              </label>
            </div>
            <Textarea
              placeholder={
                "One prompt per line…\nWhat is the capital of France?\nExplain LoRA in one sentence."
              }
              value={promptText}
              onChange={(e) => setPromptText(e.target.value)}
              rows={5}
            />
          </div>

          {/* Model selection */}
          <div className="space-y-1.5">
            <label className="text-sm font-medium">
              Models
              {selectedIds.size > 0 && (
                <span className="ml-2 rounded-full bg-primary/10 px-2 py-0.5 text-xs text-primary">
                  {selectedIds.size}
                </span>
              )}
            </label>
            <div className="max-h-48 overflow-y-auto rounded-md border border-border divide-y divide-border/50">
              {allEntries.length === 0 ? (
                <p className="px-3 py-4 text-center text-sm text-muted-foreground">
                  No models available. Load a model first.
                </p>
              ) : (
                allEntries.map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => toggle(m.id)}
                    className={cn(
                      "flex w-full items-center gap-3 px-3 py-2 text-sm transition-colors hover:bg-accent",
                      selectedIds.has(m.id) && "bg-primary/5",
                    )}
                  >
                    <span
                      className={cn(
                        "flex size-4 shrink-0 items-center justify-center rounded border transition-colors",
                        selectedIds.has(m.id)
                          ? "border-primary bg-primary text-primary-foreground"
                          : "border-border",
                      )}
                    >
                      {selectedIds.has(m.id) && (
                        <CheckIcon className="size-2.5" />
                      )}
                    </span>
                    <span className="flex-1 truncate text-left">{m.name}</span>
                    <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
                      {m.tag}
                    </span>
                  </button>
                ))
              )}
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleStart} disabled={!canStart}>
            Start Benchmark
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
