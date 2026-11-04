// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Paste or drop recipe JSON (Recipe Studio's "Copy recipe", a learning recipe file, a group exported from Flows).

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useNativeFileDrop } from "@/features/native-intents/use-native-file-drop";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useRef, useState } from "react";
import { importRecipe } from "../api";
import { FileImportIcon } from "../editor/icon-data";
import type { FlowRecord } from "../types";
import { parseRecipeJson } from "./recipe-io";

export function RecipeImportDialog({
  open,
  onOpenChange,
  onImported,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onImported: (flow: FlowRecord) => void;
}): ReactElement {
  const [text, setText] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState<{
    flow: FlowRecord;
    notices: string[];
  } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open) {
      setText("");
      setName("");
      setError(null);
      setDone(null);
    }
  }, [open]);

  const readFile = async (file: File) => {
    setText(await file.text());
    if (!name) setName(file.name.replace(/(\.recipe)?\.json$/i, ""));
    setError(null);
  };
  const {
    ref: dropRef,
    dragging,
    dragHandlers,
  } = useNativeFileDrop({
    accept: ".json",
    multiple: false,
    disabled: busy,
    onFiles: (files) => {
      const file = files[0];
      if (file) return readFile(file);
    },
  });

  const submit = async () => {
    const parsed = parseRecipeJson(text);
    if ("error" in parsed) {
      setError(parsed.error);
      return;
    }
    setBusy(true);
    try {
      const result = await importRecipe({
        payload: parsed.payload,
        ...(name.trim() && { name: name.trim() }),
      });
      setDone({ flow: result.flow, notices: result.report.notices });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Import failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg" data-testid="recipe-import-dialog">
        <DialogHeader>
          <DialogTitle>
            {done ? `Imported “${done.flow.name}”` : "Import recipe JSON"}
          </DialogTitle>
          <DialogDescription>
            {done
              ? "The recipe is a recipe group in a new flow. Nothing ran."
              : "Paste recipe JSON from Data Recipes (Copy recipe) or from a recipe group, or drop a .json file. It becomes a new flow with one recipe group."}
          </DialogDescription>
        </DialogHeader>
        {done ? (
          <div className="flex flex-col gap-2">
            {done.notices.map((line) => (
              <p
                key={line}
                className="text-amber-600 text-sm dark:text-amber-400"
              >
                {line}
              </p>
            ))}
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              ref={dropRef}
              {...dragHandlers}
              className={`flex items-center justify-center gap-2 rounded-xl border-2 border-dashed px-4 py-5 text-sm transition-colors ${
                dragging
                  ? "border-primary bg-primary/5"
                  : "border-border hover:bg-muted/50"
              }`}
              disabled={busy}
            >
              <HugeiconsIcon
                icon={FileImportIcon}
                size={20}
                className="text-muted-foreground"
              />
              Drop a recipe .json here or click to choose
            </button>
            <input
              ref={fileRef}
              type="file"
              accept=".json,application/json"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                e.target.value = "";
                if (file) void readFile(file);
              }}
            />
            <textarea
              value={text}
              onChange={(e) => {
                setText(e.target.value);
                setError(null);
              }}
              placeholder='{"recipe": {"columns": [ … ]}, "run": {…}, "ui": {…}}'
              rows={8}
              className="rounded-md border bg-transparent px-2 py-1.5 font-mono text-xs"
              data-testid="recipe-import-text"
            />
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Name (optional)"
              className="rounded-md border bg-transparent px-2 py-1.5 text-sm"
            />
            {error && (
              <p className="text-destructive text-sm" role="alert">
                {error}
              </p>
            )}
          </div>
        )}
        <DialogFooter>
          {done ? (
            <Button onClick={() => onImported(done.flow)}>Open flow</Button>
          ) : (
            <Button
              disabled={busy || !text.trim()}
              onClick={() => void submit()}
            >
              {busy ? "Importing…" : "Import recipe"}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
