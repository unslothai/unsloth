// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import type { KnowledgeBase } from "@/features/rag/api/rag-api";
import { KBCreateDialog } from "@/features/rag/components/kb-create-dialog";
import { KBDetailPanel } from "@/features/rag/components/kb-detail-panel";
import { KBList, type KBPanel } from "@/features/rag/components/kb-list";
import { PreviewPanel } from "@/features/rag/components/preview-panel";
import { RagDefaultsSection } from "@/features/rag/components/rag-defaults-section";
import { ThreadIndexList } from "@/features/rag/components/thread-index-list";
import { useResizablePanelWidth } from "@/features/rag/hooks/use-resizable-width";
import { usePreviewStore } from "@/features/rag/stores/preview-store";
import { cn } from "@/lib/utils";
import { Add01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type CSSProperties, useState } from "react";

// Height of the master-detail workspace, only rendered once a KB's upload or
// files panel is open (or a preview is active) so the section stays compact
// when just browsing the list.
const KB_WORKSPACE_HEIGHT = "h-[360px]";

export function KnowledgeBasesTab() {
  const [activeKb, setActiveKb] = useState<KnowledgeBase | null>(null);
  const [activePanel, setActivePanel] = useState<KBPanel | null>(null);
  const [createOpen, setCreateOpen] = useState(false);
  const previewTarget = usePreviewStore((s) => s.target);
  const previewStatus = usePreviewStore((s) => s.status);
  const closePreview = usePreviewStore((s) => s.close);
  const previewActive =
    previewTarget !== null ||
    previewStatus === "loading" ||
    previewStatus === "error";

  const closePanel = () => {
    setActiveKb(null);
    setActivePanel(null);
    closePreview();
  };

  const handlePanel = (kb: KnowledgeBase, panel: KBPanel) => {
    // A stale preview from a previous selection would otherwise linger and
    // force the workspace wider; clear it whenever the panel changes.
    closePreview();
    // Re-clicking the active KB's active button collapses back to full width.
    if (activeKb?.id === kb.id && activePanel === panel) {
      closePanel();
    } else {
      setActiveKb(kb);
      setActivePanel(panel);
    }
  };

  const panelOpen = activeKb !== null && activePanel !== null;
  const splitOpen = panelOpen || previewActive;

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
    // pr-2.5 insets right-aligned counts (e.g. "3 total", "60 threads") from
    // the scroll container's scrollbar, which overlays the content edge in
    // webviews that don't honor scrollbar-gutter.
    <div className="flex min-w-0 flex-col gap-4 pr-2.5">
      <div>
        <h2 className="text-lg font-semibold">Knowledge bases</h2>
        <p className="text-sm text-muted-foreground">
          Create reusable document collections and pick one per chat thread to
          ground answers in your own files.
        </p>
      </div>
      <Separator />

      <div className="flex min-w-0 flex-col gap-2">
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
        <p className="text-xs text-muted-foreground">
          Create a knowledge base and pick its defaults. Use the upload and
          files buttons on a base to add or manage its documents.
        </p>
        <div
          className={cn(
            "flex min-h-0 min-w-0 gap-4",
            splitOpen && KB_WORKSPACE_HEIGHT,
          )}
        >
          <div className={splitOpen ? "w-[260px] shrink-0" : "min-w-0 flex-1"}>
            <KBList
              activeKbId={activeKb?.id ?? null}
              activePanel={activePanel}
              onPanel={handlePanel}
              onDeleted={(id) => {
                if (activeKb?.id === id) closePanel();
              }}
            />
          </div>
          {panelOpen && activeKb && activePanel ? (
            <>
              <Separator orientation="vertical" className="h-auto" />
              {/* overflow-hidden so long document names (which spill past the
                  flex slot before truncation kicks in) can never widen the
                  workspace and shift the whole dialog right. */}
              <div className="min-w-0 flex-1 overflow-hidden">
                <KBDetailPanel
                  kb={activeKb}
                  panel={activePanel}
                  onClose={closePanel}
                />
              </div>
            </>
          ) : null}
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
                  // shrink + min-w-0 + a width cap so the preview can never
                  // push the workspace wider than the dialog (which clipped
                  // the close button and right-aligned content).
                  "w-0 min-w-0 shrink overflow-hidden max-lg:hidden lg:w-[var(--preview-w)] lg:max-w-[60%]",
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
        <KBCreateDialog
          open={createOpen}
          onOpenChange={setCreateOpen}
          onCreated={(kb) => handlePanel(kb, "upload")}
        />
      </div>

      <Separator />
      <ThreadIndexList />
      <Separator />
      <RagDefaultsSection />
    </div>
  );
}
