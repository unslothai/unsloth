// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
import type { ChunkingStrategy, KBMode } from "../api/rag-api";
import { useRagStore } from "../stores/rag-store";

/** Defaults pre-fill the KB create dialog. Same (multimodal, late) rejection as create. */
export function RagDefaultsSection() {
  const defaults = useRagStore((s) => s.defaults);
  const loadDefaults = useRagStore((s) => s.loadDefaults);
  const updateDefaults = useRagStore((s) => s.updateDefaults);

  const [chunkingStrategy, setChunkingStrategy] =
    useState<ChunkingStrategy>("standard");
  const [mode, setMode] = useState<KBMode>("text");
  const [embeddingModel, setEmbeddingModel] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void loadDefaults();
  }, [loadDefaults]);

  useEffect(() => {
    if (defaults) {
      setChunkingStrategy(defaults.chunking_strategy);
      setMode(defaults.mode);
      setEmbeddingModel(defaults.embedding_model ?? "");
    }
  }, [defaults]);

  const lateDisabled = mode === "multimodal";
  const multimodalDisabled = chunkingStrategy === "late";

  const persist = (patch: {
    chunking_strategy?: ChunkingStrategy;
    mode?: KBMode;
    embedding_model?: string | null;
  }) => {
    setError(null);
    void updateDefaults(patch).catch((err) => {
      setError(err instanceof Error ? err.message : String(err));
    });
  };

  return (
    <div className="flex flex-col gap-3">
      <div>
        <h3 className="text-sm font-medium">Defaults for new knowledge bases</h3>
        <p className="text-xs text-muted-foreground">
          Pre-fills the KB create dialog. Existing KBs keep their own
          settings — use the Reconfigure button to change those.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="defaults-mode">Mode</Label>
          <Select
            value={mode}
            onValueChange={(v) => {
              const next = v as KBMode;
              setMode(next);
              persist({ mode: next });
            }}
          >
            <SelectTrigger id="defaults-mode">
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
                Multimodal
              </SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="defaults-strategy">Chunking strategy</Label>
          <Select
            value={chunkingStrategy}
            onValueChange={(v) => {
              const next = v as ChunkingStrategy;
              setChunkingStrategy(next);
              persist({ chunking_strategy: next });
            }}
          >
            <SelectTrigger id="defaults-strategy">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="standard">Standard</SelectItem>
              <SelectItem
                value="late"
                disabled={lateDisabled}
                title={
                  lateDisabled
                    ? "Late chunking cannot be combined with multimodal mode"
                    : undefined
                }
              >
                Late chunking
              </SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="defaults-model">
          Default embedding model override (optional)
        </Label>
        <Input
          id="defaults-model"
          value={embeddingModel}
          onChange={(e) => setEmbeddingModel(e.target.value)}
          onBlur={() => persist({ embedding_model: embeddingModel })}
          placeholder="Leave blank to use the matrix default"
        />
      </div>
      {error ? <div className="text-xs text-destructive">{error}</div> : null}
    </div>
  );
}
