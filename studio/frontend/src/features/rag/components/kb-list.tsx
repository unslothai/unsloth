// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { Delete02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { KnowledgeBase } from "../api/rag-api";
import { useKnowledgeBases } from "../hooks/use-knowledge-bases";

export function KBList({
  selectedId,
  onSelect,
}: {
  selectedId: string | null;
  onSelect: (kb: KnowledgeBase | null) => void;
}) {
  const { knowledgeBases, loading, error, deleteKB } = useKnowledgeBases();

  return (
    <div className="flex flex-col gap-1">
      {error ? <div className="text-xs text-destructive">{error}</div> : null}

      {/* Capped, content-sized list: stays short when empty (no oddly-tall
          box) and scrolls internally once there are many bases. */}
      <ScrollArea className="max-h-[320px]">
        <div className="flex flex-col gap-1 pr-2">
          {knowledgeBases.length === 0 && !loading ? (
            <div className="rounded-md border border-dashed border-border/60 px-3 py-6 text-center text-xs text-muted-foreground">
              No knowledge bases yet.
            </div>
          ) : null}
          {knowledgeBases.map((kb) => {
            const isSelected = kb.id === selectedId;
            return (
              <div
                key={kb.id}
                className={cn(
                  "group flex items-center justify-between gap-2 rounded-md px-3 py-2 cursor-pointer transition-colors",
                  isSelected
                    ? "bg-accent text-accent-foreground"
                    : "hover:bg-accent/50",
                )}
                onClick={() => onSelect(kb)}
              >
                <div className="flex min-w-0 flex-col">
                  <span className="flex min-w-0 items-center gap-1.5 truncate text-sm font-medium">
                    <span className="truncate">{kb.name}</span>
                    {kb.chunking_strategy === "late" ? (
                      <span
                        className="shrink-0 rounded-sm bg-amber-500/15 px-1 text-[10px] font-medium text-amber-700 dark:text-amber-300"
                        title="Late chunking enabled"
                      >
                        ⚡ Late
                      </span>
                    ) : null}
                    {kb.mode === "multimodal" ? (
                      <span
                        className="shrink-0 rounded-sm bg-violet-500/15 px-1 text-[10px] font-medium text-violet-700 dark:text-violet-300"
                        title="Multimodal — text + image embeddings"
                      >
                        🖼️ MM
                      </span>
                    ) : null}
                  </span>
                  {kb.description ? (
                    <span className="truncate text-xs text-muted-foreground">
                      {kb.description}
                    </span>
                  ) : null}
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Delete knowledge base"
                  className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (
                      window.confirm(
                        `Delete "${kb.name}" and all its documents?`,
                      )
                    ) {
                      void deleteKB(kb.id).then(() => {
                        if (isSelected) onSelect(null);
                      });
                    }
                  }}
                >
                  <HugeiconsIcon icon={Delete02Icon} size={14} />
                </Button>
              </div>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
}
