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
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useEffect, useState } from "react";
import type {
  ChunkingStrategy,
  KBMode,
  KnowledgeBase,
} from "../api/rag-api";
import { useRagStore } from "../stores/rag-store";

export function KBReconfigureDialog({
  open,
  onOpenChange,
  kb,
  documentCount,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  kb: KnowledgeBase;
  documentCount: number;
}) {
  const reingestKB = useRagStore((s) => s.reingestKB);
  const [chunkingStrategy, setChunkingStrategy] = useState<ChunkingStrategy>(
    kb.chunking_strategy,
  );
  const [mode, setMode] = useState<KBMode>(kb.mode);
  const [embeddingModel, setEmbeddingModel] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Re-sync when the dialog opens against a different KB.
  useEffect(() => {
    if (open) {
      setChunkingStrategy(kb.chunking_strategy);
      setMode(kb.mode);
      setEmbeddingModel("");
      setError(null);
      setSubmitting(false);
    }
  }, [open, kb.id, kb.chunking_strategy, kb.mode]);

  const lateDisabled = mode === "multimodal";
  const multimodalDisabled = chunkingStrategy === "late";

  const placeholderEmbedder =
    mode === "multimodal"
      ? `Current: ${kb.embedding_model}`
      : chunkingStrategy === "late"
        ? `Current: ${kb.embedding_model}`
        : `Current: ${kb.embedding_model}`;

  const changedSettings =
    chunkingStrategy !== kb.chunking_strategy ||
    mode !== kb.mode ||
    embeddingModel.trim() !== "";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (submitting) return;
    const verb = changedSettings ? "Reconfigure and re-index" : "Re-index";
    if (
      !window.confirm(
        `${verb} ${documentCount} document${documentCount === 1 ? "" : "s"}? ` +
          `Existing chunks will be deleted and rebuilt from the original files. ` +
          `Search will be unavailable until ingestion finishes.`,
      )
    ) {
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await reingestKB(kb.id, {
        chunking_strategy: chunkingStrategy,
        mode,
        embedding_model: embeddingModel.trim() || undefined,
      });
      onOpenChange(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <form onSubmit={handleSubmit}>
          <DialogHeader>
            <DialogTitle>Reconfigure “{kb.name}”</DialogTitle>
            <DialogDescription>
              Change the chunking strategy, mode, or embedder for this KB.
              All {documentCount} document{documentCount === 1 ? "" : "s"}{" "}
              will be re-ingested from the originals on disk.
            </DialogDescription>
          </DialogHeader>
          <div className="flex flex-col gap-4 py-4">
            <div className="flex flex-col gap-2">
              <Label htmlFor="reconf-mode">Mode</Label>
              <Select
                value={mode}
                onValueChange={(v) => setMode(v as KBMode)}
              >
                <SelectTrigger id="reconf-mode">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="text">Text only</SelectItem>
                  <SelectItem
                    value="multimodal"
                    disabled={multimodalDisabled}
                    title={
                      multimodalDisabled
                        ? "Multimodal cannot be combined with late chunking"
                        : undefined
                    }
                  >
                    Multimodal — text + images
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="reconf-strategy">Chunking strategy</Label>
              <Select
                value={chunkingStrategy}
                onValueChange={(v) => setChunkingStrategy(v as ChunkingStrategy)}
              >
                <SelectTrigger id="reconf-strategy">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="standard">
                    Standard — heading-aware recursive splitter
                  </SelectItem>
                  <SelectItem
                    value="late"
                    disabled={lateDisabled}
                    title={
                      lateDisabled
                        ? "Late chunking cannot be combined with multimodal mode"
                        : undefined
                    }
                  >
                    Late chunking — single-pass embedder
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="reconf-model">Embedding model (optional)</Label>
              <Input
                id="reconf-model"
                value={embeddingModel}
                onChange={(e) => setEmbeddingModel(e.target.value)}
                placeholder={placeholderEmbedder}
              />
              <p className="text-[11px] text-muted-foreground">
                Leave blank to keep the current model (or pick the matrix
                default when mode/strategy changes).
              </p>
            </div>
            {error ? (
              <div className="text-xs text-destructive">{error}</div>
            ) : null}
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="ghost"
              onClick={() => onOpenChange(false)}
              disabled={submitting}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={submitting}>
              {submitting
                ? "Re-indexing…"
                : changedSettings
                  ? "Reconfigure & re-index"
                  : "Re-index"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
