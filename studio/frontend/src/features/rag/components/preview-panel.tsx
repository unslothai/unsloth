// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { LoaderIcon, XIcon } from "lucide-react";
import {
  type FC,
  type ReactNode,
  useEffect,
  useSyncExternalStore,
} from "react";
import type { PreviewTarget } from "../api/rag-api";
import {
  type PreviewLoadStatus,
  isInlineBlobAllowed,
  usePreviewStore,
} from "../stores/preview-store";
import { PreviewPdfView } from "./preview-pdf-view";
import { PreviewTextView } from "./preview-text-view";
import { PreviewUnavailable } from "./preview-unavailable";

interface PreviewPanelProps {
  /** Whether the panel is currently being shown in its host slot.
   *  When the host hides the slot (e.g. the user closes both
   *  settings and preview from the chat header), we run the close
   *  side-effect so the blob URL is revoked. */
  open: boolean;
  disableDrawer?: boolean;
}

const LG_BREAKPOINT = 1024;
const MEDIA_QUERY = `(max-width: ${LG_BREAKPOINT - 1}px)`;

function getLgSnapshot(): boolean {
  if (
    typeof window === "undefined" ||
    typeof window.matchMedia !== "function"
  ) {
    return false;
  }
  return window.matchMedia(MEDIA_QUERY).matches;
}

function lgSubscribe(callback: () => void): () => void {
  if (
    typeof window === "undefined" ||
    typeof window.matchMedia !== "function"
  ) {
    return () => undefined;
  }
  const mql = window.matchMedia(MEDIA_QUERY);
  mql.addEventListener("change", callback);
  return () => mql.removeEventListener("change", callback);
}

function useIsViewportSqueezed(): boolean {
  return useSyncExternalStore(lgSubscribe, getLgSnapshot, () => false);
}

interface PreviewBodyArgs {
  error: string | null;
  previewBlob: Blob | null;
  previewFileUrl: string | null;
  status: PreviewLoadStatus;
  target: PreviewTarget | null;
}

function renderPreviewBody({
  error,
  previewBlob,
  previewFileUrl,
  status,
  target,
}: PreviewBodyArgs): ReactNode {
  if (status === "loading") {
    return (
      <div className="flex h-full items-center justify-center gap-2 text-xs text-muted-foreground">
        <LoaderIcon className="size-3.5 animate-spin" />
        Loading preview…
      </div>
    );
  }

  if (status === "error") {
    // Treat 404s as "document missing" so a stale citation reads
    // like "no longer available" rather than a generic error.
    const isMissing = (error ?? "").toLowerCase().includes("not found");
    return (
      <PreviewUnavailable
        filename={target?.filename}
        reason={error ?? "Preview unavailable."}
        variant={isMissing ? "missing" : "error"}
      />
    );
  }

  if (status === "ready" && target) {
    const pdfFile = previewFileUrl ?? previewBlob;
    if (
      target.mediaKind === "pdf" &&
      pdfFile &&
      isInlineBlobAllowed(target.mediaKind)
    ) {
      return <PreviewPdfView target={target} file={pdfFile} />;
    }

    // text / image / docx / html / unknown — all routed through
    // text-view. text gets the snippet rendered inline; docx/html
    // /unknown skip inline-render entirely (contracts §5.4 + Risk #3).
    return <PreviewTextView target={target} />;
  }

  return null;
}

/** Body-only renderer. The host slot (desktop aside or mobile sheet)
 *  is owned by the host (chat-settings panel slot, kb-detail panel,
 *  etc.) — this component is purely the content of the right slot. */
export const PreviewPanel: FC<PreviewPanelProps> = ({
  open,
  disableDrawer = false,
}) => {
  const target = usePreviewStore((s) => s.target);
  const previewBlob = usePreviewStore((s) => s.previewBlob);
  const previewFileUrl = usePreviewStore((s) => s.previewFileUrl);
  const status = usePreviewStore((s) => s.status);
  const error = usePreviewStore((s) => s.error);
  const close = usePreviewStore((s) => s.close);
  const isSqueezed = useIsViewportSqueezed() && !disableDrawer;

  // Unmount + visibility cleanup: when the panel is hidden or
  // unmounted, revoke the live object URL (contracts §5.5).
  useEffect(() => {
    if (!open) {
      close();
    }
  }, [open, close]);
  useEffect(() => {
    return () => {
      // Component truly unmounting (e.g. navigation away). Cleanup
      // anything still live.
      usePreviewStore.getState().close();
    };
  }, []);

  // Keyboard accessibility: ESC closes the preview.
  useEffect(() => {
    if (!open) {
      return;
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        close();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, close]);

  const body = renderPreviewBody({
    error,
    previewBlob,
    previewFileUrl,
    status,
    target,
  });

  const renderedContent = (
    <section
      aria-label="Document preview"
      className="flex h-full flex-col overflow-hidden bg-panel-surface/85 dark:bg-background/85 backdrop-blur-lg text-panel-surface-fg border border-border/40 shadow-lg menu-soft-surface"
    >
      <div className="flex items-center justify-between gap-2 border-b border-border/60 px-3 py-2 font-heading">
        <div className="flex items-center gap-1.5">
          <span
            className="h-2 w-2 animate-pulse rounded-full bg-primary [--pulse-color:color-mix(in_oklab,var(--primary)_35%,transparent)]"
            aria-hidden="true"
          />
          <span className="text-sm font-semibold">Preview</span>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={close}
          aria-label="Close preview"
          className="size-7"
        >
          <XIcon className="size-3.5" />
        </Button>
      </div>
      <div className="min-h-0 flex-1 overflow-hidden">{body}</div>
    </section>
  );

  if (isSqueezed) {
    return (
      <>
        <div className="hidden" aria-hidden="true" />
        <Sheet
          open={open}
          onOpenChange={(next) => {
            if (!next) {
              close();
            }
          }}
        >
          <SheetContent
            side="right"
            showCloseButton={false}
            overlayClassName="bg-background/35 supports-backdrop-filter:backdrop-blur-[1px]"
            className="preview-sheet-content p-0 font-heading data-[side=right]:w-full data-[side=right]:sm:max-w-md"
          >
            <SheetHeader className="sr-only">
              <SheetTitle>Document preview</SheetTitle>
              <SheetDescription>
                Preview of the active document citation
              </SheetDescription>
            </SheetHeader>
            <div className="flex h-full flex-col">{renderedContent}</div>
          </SheetContent>
        </Sheet>
      </>
    );
  }

  return renderedContent;
};
