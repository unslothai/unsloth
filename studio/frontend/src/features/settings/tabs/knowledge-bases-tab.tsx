// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Separator } from "@/components/ui/separator";
import type { KnowledgeBase } from "@/features/rag/api/rag-api";
import { KBDetailPanel } from "@/features/rag/components/kb-detail-panel";
import { KBList } from "@/features/rag/components/kb-list";
import { PreviewPanel } from "@/features/rag/components/preview-panel";
import { RagDefaultsSection } from "@/features/rag/components/rag-defaults-section";
import { ThreadIndexList } from "@/features/rag/components/thread-index-list";
import { useResizablePanelWidth } from "@/features/rag/hooks/use-resizable-width";
import { usePreviewStore } from "@/features/rag/stores/preview-store";
import { cn } from "@/lib/utils";
import { type CSSProperties, useState } from "react";

export function KnowledgeBasesTab() {
  const [selected, setSelected] = useState<KnowledgeBase | null>(null);
  const previewTarget = usePreviewStore((s) => s.target);
  const previewStatus = usePreviewStore((s) => s.status);
  const previewActive =
    previewTarget !== null ||
    previewStatus === "loading" ||
    previewStatus === "error";

  const {
    width: previewWidth,
    isResizing: previewResizing,
    startResize: startPreviewResize,
    adjustWidth: adjustPreviewWidth,
    resetWidth: resetPreviewWidth,
  } = useResizablePanelWidth({
    storageKey: "unsloth:kb-preview-panel-width",
    defaultWidth: 560,
    minWidth: 320,
    maxWidthFraction: 0.7,
    enabled: previewActive,
  });

  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      <div>
        <h2 className="text-lg font-semibold">Knowledge bases</h2>
        <p className="text-sm text-muted-foreground">
          Create reusable document collections and pick one per chat thread to
          ground answers in your own files.
        </p>
      </div>
      <Separator />
      <div className="flex min-h-0 flex-1 gap-4">
        <div className="w-[220px] shrink-0">
          <KBList selectedId={selected?.id ?? null} onSelect={setSelected} />
        </div>
        <Separator orientation="vertical" />
        <div className="min-w-0 flex-1">
          {selected ? (
            <KBDetailPanel kb={selected} />
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
              Select a knowledge base, or create a new one to get started.
            </div>
          )}
        </div>
        {previewActive ? (
          <>
            <button
              type="button"
              aria-label="Resize preview panel — drag, or use arrow keys"
              aria-orientation="vertical"
              onPointerDown={startPreviewResize}
              onDoubleClick={resetPreviewWidth}
              onKeyDown={(event) => {
                if (event.key === "ArrowLeft") {
                  event.preventDefault();
                  adjustPreviewWidth(24);
                } else if (event.key === "ArrowRight") {
                  event.preventDefault();
                  adjustPreviewWidth(-24);
                } else if (event.key === "Home") {
                  event.preventDefault();
                  resetPreviewWidth();
                }
              }}
              className={cn(
                "group relative hidden w-1.5 cursor-col-resize touch-none select-none border-0 bg-transparent p-0 outline-none lg:block",
                "before:absolute before:inset-y-0 before:left-1/2 before:-translate-x-1/2 before:w-px before:bg-border/70 before:transition-colors",
                "hover:before:bg-primary/40 focus-visible:before:bg-primary/60",
                previewResizing && "before:bg-primary/60",
              )}
            >
              <span
                aria-hidden="true"
                className={cn(
                  "absolute left-1/2 top-1/2 h-10 w-[3px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-border opacity-0 transition-opacity",
                  "group-hover:opacity-100 group-focus-visible:opacity-100",
                  previewResizing && "opacity-100",
                )}
              />
            </button>
            <div
              className={cn(
                "w-0 shrink-0 overflow-hidden max-lg:hidden lg:w-[var(--preview-w)]",
                !previewResizing && "transition-[width] duration-200 ease-out",
              )}
              style={
                {
                  "--preview-w": `${previewWidth}px`,
                } as CSSProperties
              }
            >
              <PreviewPanel open={previewActive} />
            </div>
          </>
        ) : null}
      </div>
      <Separator />
      <ThreadIndexList />
      <Separator />
      <RagDefaultsSection />
    </div>
  );
}
