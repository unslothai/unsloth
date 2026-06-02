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
import { useEffect, useState } from "react";
import type { KnowledgeBase } from "../api/rag-api";
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

  const initialEmbedder = defaults?.embedding_model ?? "";

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [embeddingModel, setEmbeddingModel] = useState(initialEmbedder);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open && !defaults) {
      void loadDefaults();
    }
  }, [open, defaults, loadDefaults]);

  useEffect(() => {
    if (open && defaults) {
      setEmbeddingModel(defaults.embedding_model ?? "");
    }
    // Only on open-flip, not on every defaults change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const reset = () => {
    setName("");
    setDescription("");
    setEmbeddingModel(defaults?.embedding_model ?? "");
    setError(null);
    setSubmitting(false);
  };

  const placeholderEmbedder = "Defaults to BAAI/bge-small-en-v1.5";

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
