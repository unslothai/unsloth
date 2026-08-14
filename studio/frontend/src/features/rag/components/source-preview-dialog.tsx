// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import { MarkdownPreview } from "@/components/markdown/markdown-preview";
import { ArtifactHtmlFrame } from "@/features/chat/artifacts/html-frame";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { useCallback, useEffect, useState } from "react";
import {
  getDocumentContent,
  getDocumentFileUrl,
  updateDocumentContent,
} from "../api/rag-api";
import type { DocumentContent } from "../types/rag";
import { PdfPreview } from "./document-preview-sheet";

/** Click-to-open viewer and quick editor for one project source.
 *
 * A modal rather than a side panel because the project page already docks Run
 * settings on the right. Escape, a backdrop click and the corner X all resolve
 * through Radix's single `onOpenChange`, so the unsaved-changes guard below is
 * one branch covering all three.
 *
 * What is editable is decided by the backend (`editable` / `readOnlyReason`), not
 * here: PDFs and Word documents have no faithful plain-text round trip, and
 * linked-folder sources would lose an edit to the next folder sync. */
export function SourcePreviewDialog({
  documentId,
  filename,
  onClose,
  onSaved,
}: {
  /** Non-null opens the modal. */
  documentId: string | null;
  filename: string;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [content, setContent] = useState<DocumentContent | null>(null);
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [editing, setEditing] = useState(false);
  const [confirmingDiscard, setConfirmingDiscard] = useState(false);

  const open = documentId !== null;

  useEffect(() => {
    if (!open || !documentId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setContent(null);
    setFileUrl(null);
    setText("");
    setEditing(false);
    setConfirmingDiscard(false);
    (async () => {
      try {
        const loaded = await getDocumentContent(documentId);
        if (cancelled) return;
        setContent(loaded);
        setText(loaded.text ?? "");
        if (loaded.mediaKind === "pdf") {
          const url = await getDocumentFileUrl(documentId);
          if (!cancelled) setFileUrl(url);
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [open, documentId]);

  const dirty = content?.editable === true && text !== (content.text ?? "");

  // Escape, backdrop and X all arrive here. Closing is refused while dirty so the
  // confirmation can decide, and refused while saving so the request that is about
  // to swap this document out is not abandoned half way.
  const handleOpenChange = useCallback(
    (next: boolean) => {
      if (next) return;
      if (saving) return;
      if (dirty) {
        setConfirmingDiscard(true);
        return;
      }
      onClose();
    },
    [dirty, saving, onClose],
  );

  async function handleSave() {
    if (!documentId || !dirty) return;
    setSaving(true);
    try {
      await updateDocumentContent(documentId, text);
      onSaved();
      onClose();
    } catch (e) {
      // Keep the modal open with the typing intact so the edit can be retried.
      toast.error(e instanceof Error ? e.message : "Could not save this source");
    } finally {
      setSaving(false);
    }
  }

  // "source" has nothing to switch to, so those open straight in the editor.
  const hasView = content !== null && content.preview !== "source";
  const showEditor = content?.editable === true && (!hasView || editing);

  return (
    <>
      <Dialog open={open} onOpenChange={handleOpenChange}>
        <DialogContent className="flex h-[80dvh] w-full flex-col gap-4 sm:max-w-3xl">
          <DialogHeader className="pr-10">
            <DialogTitle className="truncate text-base">{filename}</DialogTitle>
            {/* Why a source is read-only is stated once, in the footer. Repeating
              it here put the same sentence twice on screen. */}
            <DialogDescription>
              {content?.editable
                ? "Edits are saved to this project's copy and the source is re-indexed."
                : "Preview of this source."}
            </DialogDescription>
          </DialogHeader>

          {hasView && content?.editable ? (
            <div className="flex shrink-0 gap-1">
              {(["view", "edit"] as const).map((mode) => {
                const active = (mode === "edit") === editing;
                return (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => setEditing(mode === "edit")}
                    className={cn(
                      "rounded-full px-3 py-1 text-ui-11 capitalize transition-colors",
                      active
                        ? "bg-muted font-medium text-foreground"
                        : "text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {mode}
                  </button>
                );
              })}
            </div>
          ) : null}

          <div className="min-h-0 flex-1 overflow-hidden">
            {loading ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Spinner className="size-3.5" /> Loading source…
              </div>
            ) : error ? (
              <p className="text-sm text-muted-foreground">
                Could not open this source ({error}).
              </p>
            ) : content?.mediaKind === "pdf" ? (
              fileUrl ? (
                <PdfPreview fileUrl={fileUrl} initialPage={1} regions={[]} />
              ) : (
                <p className="text-sm text-muted-foreground">
                  This source has no file to preview.
                </p>
              )
            ) : showEditor ? (
              <Textarea
                value={text}
                fieldSizing="fixed"
                onChange={(e) => setText(e.target.value)}
                disabled={saving}
                spellCheck={false}
                aria-label={`Edit ${filename}`}
                className="h-full overflow-auto font-mono text-sm leading-relaxed"
              />
            ) : content?.preview === "markdown" ? (
              <MarkdownPreview
                markdown={text}
                className="h-full max-h-none overflow-auto border-0 bg-transparent p-0 text-sm"
              />
            ) : content?.preview === "html" ? (
              // The chat artifact canvas: a sandboxed, opaque-origin iframe with
              // network access off by default. Rendering an uploaded page is only
              // safe inside it, so this must never become a plain innerHTML.
              <div className="h-full overflow-hidden rounded-xl border">
                <ArtifactHtmlFrame
                  code={text}
                  fill={true}
                  title={`${filename} preview`}
                />
              </div>
            ) : text ? (
              <pre className="h-full overflow-auto whitespace-pre-wrap break-words rounded-xl bg-muted/30 p-4 font-mono text-sm leading-relaxed text-foreground/90">
                {text}
              </pre>
            ) : (
              <p className="text-sm text-muted-foreground">
                This source has no text to preview.
              </p>
            )}
          </div>

          <DialogFooter className="shrink-0 sm:items-center sm:justify-between">
            <p className="text-ui-11 text-muted-foreground">
              {content && !content.editable ? content.readOnlyReason : null}
            </p>
            {/* Save is the only button: closing is the corner X, which runs the
              same unsaved-changes guard a Cancel would have. */}
            {content?.editable ? (
              <Button
                type="button"
                disabled={!dirty || saving}
                onClick={() => void handleSave()}
              >
                {saving ? "Saving…" : "Save"}
              </Button>
            ) : null}
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog
        open={confirmingDiscard}
        onOpenChange={(next) => {
          if (!next) setConfirmingDiscard(false);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Discard changes?</AlertDialogTitle>
            <AlertDialogDescription>
              Your edits to {filename} have not been saved.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep editing</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                setConfirmingDiscard(false);
                onClose();
              }}
            >
              Discard
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
