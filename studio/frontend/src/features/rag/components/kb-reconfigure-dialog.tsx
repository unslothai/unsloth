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
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
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
  const [embeddingModel, setEmbeddingModel] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Re-sync when the dialog opens against a different KB.
  useEffect(() => {
    if (open) {
      setEmbeddingModel("");
      setError(null);
      setSubmitting(false);
    }
  }, [open, kb.id]);

  const placeholderEmbedder = `Current: ${kb.embedding_model}`;

  const changedSettings = embeddingModel.trim() !== "";

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
        embedding_model: embeddingModel.trim() || undefined,
        caption_images: useChatRuntimeStore.getState().ragCaptionImages,
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
              Change the embedder for this KB.
              All {documentCount} document{documentCount === 1 ? "" : "s"}{" "}
              will be re-ingested from the originals on disk.
            </DialogDescription>
          </DialogHeader>
          <div className="flex flex-col gap-4 py-4">
            <div className="flex flex-col gap-2">
              <Label htmlFor="reconf-model">Embedding model (optional)</Label>
              <Input
                id="reconf-model"
                value={embeddingModel}
                onChange={(e) => setEmbeddingModel(e.target.value)}
                placeholder={placeholderEmbedder}
              />
              <p className="text-[11px] text-muted-foreground">
                Leave blank to keep the current model (or pick the default
                when the mode changes).
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
