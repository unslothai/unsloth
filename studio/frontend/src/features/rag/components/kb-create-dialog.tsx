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
import { useState } from "react";
import type { ChunkingStrategy, KnowledgeBase } from "../api/rag-api";
import { useKnowledgeBases } from "../hooks/use-knowledge-bases";

export function KBCreateDialog({
  open,
  onOpenChange,
  onCreated,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreated?: (kb: KnowledgeBase) => void;
}) {
  const { createKB } = useKnowledgeBases();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [embeddingModel, setEmbeddingModel] = useState("");
  const [chunkingStrategy, setChunkingStrategy] =
    useState<ChunkingStrategy>("standard");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reset = () => {
    setName("");
    setDescription("");
    setEmbeddingModel("");
    setChunkingStrategy("standard");
    setError(null);
    setSubmitting(false);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const kb = await createKB({
        name: name.trim(),
        description: description.trim() || undefined,
        embedding_model: embeddingModel.trim() || undefined,
        chunking_strategy: chunkingStrategy,
      });
      onCreated?.(kb);
      reset();
      onOpenChange(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (!o) reset();
        onOpenChange(o);
      }}
    >
      <DialogContent>
        <form onSubmit={handleSubmit}>
          <DialogHeader>
            <DialogTitle>Create knowledge base</DialogTitle>
            <DialogDescription>
              A knowledge base groups documents you can reuse across multiple
              chat threads.
            </DialogDescription>
          </DialogHeader>
          <div className="flex flex-col gap-4 py-4">
            <div className="flex flex-col gap-2">
              <Label htmlFor="kb-name">Name</Label>
              <Input
                id="kb-name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Internal docs"
                autoFocus
                required
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="kb-description">Description (optional)</Label>
              <Input
                id="kb-description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="What's in this KB?"
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="kb-strategy">Chunking strategy</Label>
              <Select
                value={chunkingStrategy}
                onValueChange={(v) => setChunkingStrategy(v as ChunkingStrategy)}
              >
                <SelectTrigger id="kb-strategy">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="standard">
                    Standard — heading-aware recursive splitter
                  </SelectItem>
                  <SelectItem value="late">
                    Late chunking — single-pass embedder, slower ingest
                  </SelectItem>
                </SelectContent>
              </Select>
              <p className="text-[11px] text-muted-foreground">
                Late chunking embeds the whole document in one pass, so each
                chunk vector carries full-document context. Slower to ingest
                (one forward pass per doc) but improves retrieval on long,
                cross-referenced text. Cannot be combined with multimodal
                mode.
              </p>
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="kb-model">Embedding model (optional)</Label>
              <Input
                id="kb-model"
                value={embeddingModel}
                onChange={(e) => setEmbeddingModel(e.target.value)}
                placeholder={
                  chunkingStrategy === "late"
                    ? "Defaults to nomic-ai/nomic-embed-text-v1.5"
                    : "Defaults to BAAI/bge-small-en-v1.5"
                }
              />
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
            <Button type="submit" disabled={!name.trim() || submitting}>
              {submitting ? "Creating…" : "Create"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
