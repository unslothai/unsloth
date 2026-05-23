// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { Add01Icon, Delete02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import type { KnowledgeBase } from "../api/rag-api";
import { useKnowledgeBases } from "../hooks/use-knowledge-bases";
import { KBCreateDialog } from "./kb-create-dialog";

export function KBList({
  selectedId,
  onSelect,
}: {
  selectedId: string | null;
  onSelect: (kb: KnowledgeBase | null) => void;
}) {
  const { knowledgeBases, loading, error, deleteKB } = useKnowledgeBases();
  const [createOpen, setCreateOpen] = useState(false);

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium">Knowledge bases</h3>
        <Button
          variant="ghost"
          size="icon"
          aria-label="Create knowledge base"
          onClick={() => setCreateOpen(true)}
        >
          <HugeiconsIcon icon={Add01Icon} size={16} />
        </Button>
      </div>

      {error ? <div className="text-xs text-destructive">{error}</div> : null}

      <ScrollArea className="flex-1">
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
                  <span className="truncate text-sm font-medium">{kb.name}</span>
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

      <KBCreateDialog
        open={createOpen}
        onOpenChange={setCreateOpen}
        onCreated={(kb) => onSelect(kb)}
      />
    </div>
  );
}
