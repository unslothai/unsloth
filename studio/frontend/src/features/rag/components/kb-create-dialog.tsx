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
import { useKnowledgeBases } from "../hooks/use-knowledge-bases";
import { useRagStore } from "../stores/rag-store";

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
  const defaults = useRagStore((s) => s.defaults);
  const loadDefaults = useRagStore((s) => s.loadDefaults);

  const initialStrategy: ChunkingStrategy =
    defaults?.chunking_strategy ?? "standard";
  const initialMode: KBMode = defaults?.mode ?? "text";
  const initialEmbedder = defaults?.embedding_model ?? "";

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [embeddingModel, setEmbeddingModel] = useState(initialEmbedder);
  const [chunkingStrategy, setChunkingStrategy] =
    useState<ChunkingStrategy>(initialStrategy);
  const [mode, setMode] = useState<KBMode>(initialMode);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open && !defaults) {
      void loadDefaults();
    }
  }, [open, defaults, loadDefaults]);

  useEffect(() => {
    if (open && defaults) {
      setChunkingStrategy(defaults.chunking_strategy);
      setMode(defaults.mode);
      setEmbeddingModel(defaults.embedding_model ?? "");
    }
    // Only on open-flip, not on every defaults change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const reset = () => {
    setName("");
    setDescription("");
    setEmbeddingModel(defaults?.embedding_model ?? "");
    setChunkingStrategy(defaults?.chunking_strategy ?? "standard");
    setMode(defaults?.mode ?? "text");
    setError(null);
    setSubmitting(false);
  };

  // Forbid (multimodal, late); disable the other side when one is picked.
  const lateDisabled = mode === "multimodal";
  const multimodalDisabled = chunkingStrategy === "late";

  const placeholderEmbedder =
    mode === "multimodal"
      ? "Defaults to BAAI/BGE-VL-base"
      : chunkingStrategy === "late"
        ? "Defaults to nomic-ai/nomic-embed-text-v1.5"
        : "Defaults to BAAI/bge-small-en-v1.5";

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
        mode,
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
              <Label htmlFor="kb-mode">Mode</Label>
              <Select
                value={mode}
                onValueChange={(v) => setMode(v as KBMode)}
              >
                <SelectTrigger id="kb-mode">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="text">
                    Text only — embed text chunks
                  </SelectItem>
                  <SelectItem
                    value="multimodal"
                    disabled={multimodalDisabled}
                    title={
                      multimodalDisabled
                        ? "Multimodal cannot be combined with late chunking"
                        : undefined
                    }
                  >
                    Multimodal — also embed images alongside text
                  </SelectItem>
                </SelectContent>
              </Select>
              <p className="text-[11px] text-muted-foreground">
                Multimodal mode extracts figures from your documents and
                embeds them in a shared text + image vector space (BGE-VL),
                so retrieval can match visual content. Larger embedder
                (~1.5&nbsp;GB VRAM) and cannot be combined with late
                chunking.
              </p>
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
                  <SelectItem
                    value="late"
                    disabled={lateDisabled}
                    title={
                      lateDisabled
                        ? "Late chunking cannot be combined with multimodal mode"
                        : undefined
                    }
                  >
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
                placeholder={placeholderEmbedder}
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
