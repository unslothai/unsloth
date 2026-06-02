// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { FilesIcon, Trash2Icon, UploadIcon } from "lucide-react";
import type { KnowledgeBase } from "../api/rag-api";
import { useKnowledgeBases } from "../hooks/use-knowledge-bases";

export type KBPanel = "upload" | "files";

export function KBList({
  activeKbId,
  activePanel,
  onPanel,
  onDeleted,
}: {
  activeKbId: string | null;
  activePanel: KBPanel | null;
  onPanel: (kb: KnowledgeBase, panel: KBPanel) => void;
  onDeleted: (kbId: string) => void;
}) {
  const { knowledgeBases, loading, error, deleteKB } = useKnowledgeBases();

  return (
    <div className="flex flex-col gap-1">
      {error ? <div className="text-xs text-destructive">{error}</div> : null}

      {/* Capped, content-sized list: short when empty, scrolls when long. */}
      <ScrollArea className="max-h-[320px]">
        <div className="flex flex-col gap-1 pr-2">
          {knowledgeBases.length === 0 && !loading ? (
            <div className="rounded-md border border-dashed border-border/60 px-3 py-6 text-center text-xs text-muted-foreground">
              No knowledge bases yet.
            </div>
          ) : null}
          {knowledgeBases.map((kb) => {
            const isActive = kb.id === activeKbId;
            return (
              <div
                key={kb.id}
                className={cn(
                  "group flex items-center justify-between gap-2 rounded-md border px-3 py-2 transition-colors",
                  isActive
                    ? "border-primary/50 bg-accent"
                    : "border-border/60 hover:bg-accent/50",
                )}
              >
                <div className="flex min-w-0 flex-col">
                  <span className="flex min-w-0 items-center gap-1.5 truncate text-sm font-medium">
                    <span className="truncate">{kb.name}</span>
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
                <div className="flex shrink-0 items-center gap-0.5">
                  <Button
                    variant="ghost"
                    size="icon"
                    aria-label={`Upload documents to ${kb.name}`}
                    title="Upload documents"
                    className={cn(
                      "h-7 w-7",
                      isActive && activePanel === "upload" && "text-primary",
                    )}
                    onClick={() => onPanel(kb, "upload")}
                  >
                    <UploadIcon className="size-3.5" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    aria-label={`View documents in ${kb.name}`}
                    title="View documents"
                    className={cn(
                      "h-7 w-7",
                      isActive && activePanel === "files" && "text-primary",
                    )}
                    onClick={() => onPanel(kb, "files")}
                  >
                    <FilesIcon className="size-3.5" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    aria-label={`Delete ${kb.name}`}
                    title="Delete knowledge base"
                    className="h-7 w-7 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100 hover:text-destructive"
                    onClick={() => {
                      if (
                        window.confirm(
                          `Delete "${kb.name}" and all its documents?`,
                        )
                      ) {
                        void deleteKB(kb.id).then(() => onDeleted(kb.id));
                      }
                    }}
                  >
                    <Trash2Icon className="size-3.5" />
                  </Button>
                </div>
              </div>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
}
